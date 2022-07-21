#!/usr/bin/env python
# coding: utf-8

import io
import sys
import requests
import unidecode

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from itertools import chain, product

import perkins
import perkins.requests
import perkins.input.snis


BASE_URL = 'https://reportes-siahv.minsalud.gob.bo/'
URL = BASE_URL + 'Reporte_Dinamico_Covid.aspx'
MAX_TRY = 5

COL_ADD_1 = 'ctl00_ContenidoPrincipal_pivotReporteCovid_pgHeader14'
COL_ADD_2 = 'ctl00_ContenidoPrincipal_pivotReporteCovid_pgHeader16'

COL_BEFORE_1 = 'ctl00_ContenidoPrincipal_pivotReporteCovid_sortedpgHeader13'
COL_BEFORE_2 = 'ctl00_ContenidoPrincipal_pivotReporteCovid_sortedpgHeader14'

COL_GROUP_ID = 'ctl00$ContenidoPrincipal$pivotReporteCovid'

FORM_DATA = {
    # data
    'ctl00$ContenidoPrincipal$ddl_sedes': 0,
    # button inputs
    'ctl00$ContenidoPrincipal$BTNGenerar': None,
    'ctl00$ContenidoPrincipal$ASPxButton1': None,
    'ctl00$ContenidoPrincipal$ASPxButton2': None,
    # export format
    'ContenidoPrincipal_ASPxComboBox1_VI': 1,
    'ctl00$ContenidoPrincipal$ASPxComboBox1': 'Excel',
    'ctl00$ContenidoPrincipal$ASPxComboBox1$DDD$L': 1,
}

DEPTS = {
    'chuquisaca': 1,
    'la.paz': 2,
    'cochabamba': 3,
    'oruro': 4,
    'potosi': 5,
    'tarija': 6,
    'santa.cruz': 7,
    'beni': 8,
    'pando': 9
}


AGE_MAP = {
    '0-19': ['a', 'b', 'c', 'd', 'e'],
    '20-39': ['f'],
    '40-49': ['g'],
    '50-59': ['h'],
    '>= 60': ['i']
}
CAT_AGE_MAP = dict(chain(*[product(v,(k,)) for k,v in AGE_MAP.items()]))


def parse_df(data_df):
    data_df = data_df.loc[3:]
    data_df = data_df.loc[:, ~data_df.T.isna().T.all(axis=0)]
    data_df = data_df.iloc[:,:-1]

    data_df.iloc[:2] = data_df.iloc[:2].fillna(method='ffill', axis=1)
    data_df = data_df[data_df.columns[~(
        data_df.iloc[0].astype(str).str.contains('Total').fillna(False) |
        data_df.iloc[1].str.contains('Total').fillna(False)
    )]]

    data_df.iloc[:, 0] = data_df.iloc[:, 0].fillna(method='ffill')
    data_df = data_df[~data_df.iloc[:, 0].str.contains('Total').fillna(False)]

    date_index = data_df.iloc[:2].apply(lambda _: '{}-{}'.format(_.iloc[0], _.iloc[1]))
    date_index = pd.to_datetime(date_index.iloc[2:])

    data_df = data_df.iloc[1:]
    data_df.iloc[0, 2:] = date_index

    data_df.columns = pd.MultiIndex.from_frame(data_df.iloc[:2].T)
    data_df = data_df.iloc[2:]

    data_df.index = data_df.iloc[:, 0]
    data_df.index.name = 'Municipio'

    data_df = data_df.fillna(0)
    data_df = data_df.iloc[:, 1:]

    df = pd.DataFrame([])

    for muni, muni_df in data_df.groupby('Municipio'):
        muni_df.index = muni_df.iloc[:, 0]
        muni_df = muni_df.iloc[:, 1:]

        fixed_df = pd.DataFrame([])

        for age, age_df in muni_df.groupby(lambda _: CAT_AGE_MAP[_[0]]):
            fixed_df[age] = age_df.sum(axis=0)

        muni = unidecode.unidecode(muni).lower()
        fixed_df.columns = pd.MultiIndex.from_product([[muni], fixed_df.columns])

        df = pd.concat([df, fixed_df], axis=1)

    return df.T


def do_fetch_data(soup, dept_key, dept_code, cookies, proxy=None):
    soup = perkins.input.snis.process_request(URL, soup, cookies, {
        **FORM_DATA,
        '__EVENTTARGET': 'ctl00$ContenidoPrincipal$ddl_sedes',
        'ctl00$ContenidoPrincipal$ScriptManager1': (
            'ctl00$ContenidoPrincipal$UpdatePanel1|ctl00$ContenidoPrincipal$BTNGenerar'
        ),
        'ctl00$ContenidoPrincipal$BTNGenerar': 'Generar',
        'ctl00$ContenidoPrincipal$ddl_sedes': dept_code,
    }, proxy)

    fecha_registro = soup.select_one('#ContenidoPrincipal_lbl_fecha').text.split(' ')[-1]
    fecha_registro = pd.to_datetime(fecha_registro, dayfirst=True)

    content = perkins.input.snis.process_request(URL, soup, cookies, {
        **FORM_DATA,
        'ctl00$ContenidoPrincipal$ASPxButton1': '',
        'ctl00$ContenidoPrincipal$ddl_sedes': dept_code,
    }, proxy, raw=True)

    data_df = pd.read_excel(io.BytesIO(content), header=None)
    data_df = parse_df(data_df)

    data_df['departamento'] = dept_key
    data_df = data_df.set_index('departamento', append=True)

    data_df.index.names = ['municipio', 'edad', 'departamento']
    data_df = data_df.reorder_levels(['departamento', 'municipio', 'edad'])

    return fecha_registro, data_df


def fetch_data(*args, _try=0, **kwargs):
    try:
        return do_fetch_data(*args, **kwargs)

    except requests.exceptions.ConnectionError as e:
        if _try > MAX_TRY or 'proxy' not in kwargs:
            raise(e)

        kwargs['proxy'] = perkins.requests.setup_proxy(BASE_URL)
        return fetch_data(*args, _try=_try+1, **kwargs)


def setup_fields(soup, cookies, proxy):
    # Agrega columna `mesDefuncion`
    soup = perkins.input.snis.process_request(URL, soup, cookies, {
        **FORM_DATA,
        '__CALLBACKID': COL_GROUP_ID,
        '__CALLBACKPARAM': '|'.join(['c0:D', COL_ADD_1, COL_BEFORE_1, 'true'])
    }, proxy)

    # Agrega columna `Gestion`
    soup = perkins.input.snis.process_request(URL, soup, cookies, {
        **FORM_DATA,
        '__CALLBACKID': COL_GROUP_ID,
        '__CALLBACKPARAM': '|'.join(['c0:D', COL_ADD_2, COL_BEFORE_2, 'true'])
    }, proxy)

    return soup


def do_config_request():
    # Busca un proxy que funcione
    proxy = None
    if '--direct' not in sys.argv:
        proxy = perkins.requests.setup_proxy(BASE_URL)

    if proxy is None and '--direct' not in sys.argv:
        raise Exception('No proxy!')

    # Primer request
    req = perkins.requests.do_request(URL, timeout=60, proxies=proxy)
    soup = BeautifulSoup(req.content, 'html.parser')

    cookies = req.headers['Set-Cookie']
    cookies = [cookie.split(';')[0] for cookie in cookies.split(',')]

    # Agrega columnas
    soup = setup_fields(soup, cookies, proxy)

    return proxy, cookies, soup


def config_request(_try=0):
    try:
        return do_config_request()

    except requests.exceptions.ConnectionError as e:
        if _try > MAX_TRY:
            raise(e)

    return config_request(_try=_try + 1)


DATA_FILE = './hechos.vitales.covid.csv'
def merge_data(df, fecha_registro):
    siahv_df = pd.read_csv(
        DATA_FILE,
        parse_dates=['fecha_deceso', 'fecha_registro']
    )

    df = df.stack().T
    df.columns.names = df.columns.names[:3] + ['sexo']

    df = df.rename({
        'No puede determinarse': 'D',
        '[sin dato]': 'D',
        'Masculino': 'M',
        'Femenino': 'F'
    }, axis=1, level='sexo')

    df = df.groupby(df.columns.names, axis=1).sum(min_count=1)
    df.index = pd.to_datetime(df.index)

    df = df.replace(0, np.nan).dropna(how='all', axis=1).T

    siahv_diff = siahv_df.groupby(siahv_df.columns[:5].to_list())['decesos'].sum()
    siahv_diff = siahv_diff.unstack(level='fecha_deceso')

    data_index = pd.concat([
        siahv_diff.index.to_frame(),
        df.index.to_frame()
    ])
    data_index = data_index[~data_index.index.duplicated()]

    siahv_diff = siahv_diff.reindex(data_index.index).reindex(df.columns, axis=1)
    siahv_diff = siahv_diff.fillna(0)

    siahv_diff = df.reindex(data_index.index).fillna(0) - siahv_diff

    siahv_diff = siahv_diff.replace(0, np.nan).stack()
    siahv_diff = siahv_diff.rename(fecha_registro).to_frame()

    siahv_diff = siahv_diff.unstack()
    siahv_diff.columns.names = ['fecha_registro', 'fecha_deceso']
    siahv_diff = siahv_diff.stack(level=[1, 0]).rename('decesos')

    siahv_df = siahv_df.set_index(siahv_df.columns[:-1].to_list())['decesos']

    siahv_df = pd.concat([siahv_df, siahv_diff])
    siahv_df = siahv_df.sort_index()
    siahv_df = siahv_df.astype(int)

    siahv_df.to_csv(DATA_FILE)


if __name__ == '__main__':
    proxy, cookies, soup = config_request()

    # Exporta datos
    death_df = None
    for dept_key, dept_code in DEPTS.items():
        fecha_registro, data_df = fetch_data(
            soup, dept_key, dept_code, cookies, proxy=proxy
        )
        death_df = pd.concat([death_df, data_df])

    death_df.columns.names = ['', '']
    merge_data(death_df, fecha_registro)
