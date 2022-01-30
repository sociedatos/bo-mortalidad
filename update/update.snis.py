#!/usr/bin/env python
# coding: utf-8

import io
import sys
import unidecode

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from itertools import chain, product

import perkins
import perkins.requests
import perkins.input.snis


BASE_URL = 'https://estadisticas.minsalud.gob.bo'
URL = BASE_URL + '/Reportes_Dinamicos/WF_Reporte_Gral_2021.aspx'

FORMAT_COLUMNS = [
    (
        'ctl00_MainContent_ASPxPivotGrid1_sortedpgHeader1',
        'ctl00_MainContent_ASPxPivotGrid1_pgHeader2',
    ),
    (
        'ctl00_MainContent_ASPxPivotGrid1_DHP_pgHeader17',
        'ctl00_MainContent_ASPxPivotGrid1_pgHeader1',
    ),
    (
        'ctl00_MainContent_ASPxPivotGrid1_pgHeader4',
        'ctl00_MainContent_ASPxPivotGrid1_sortedpgHeader14',
    ),
    (
        'ctl00_MainContent_ASPxPivotGrid1_pgHeader13',
        'ctl00_MainContent_ASPxPivotGrid1_pgArea1',
    )
]

FORM_DATA = {
    '__EVENTTARGET': '',
    'ctl00$MainContent$DDL_sedes': 1,
    'ctl00$MainContent$DDL_form': '302',
    'ctl00$MainContent$DDL_var': '09',
    'ctl00$MainContent$ASPxPivotGrid1$DXHFP$TPCFCm1$O': None,
    'ctl00$MainContent$ASPxPivotGrid1$DXHFP$TPCFCm1$C': None,
    'ctl00$MainContent$Button1': None,
    'ctl00$MainContent$ASPxButton1': None,
    'MainContent_ASPxComboBox1_VI': 1,
    'ctl00$MainContent$ASPxComboBox1': 'Excel',
    'ctl00$MainContent$ASPxComboBox1$DDD$L': 1,
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
    '0-19': ['a', 'b', 'c', 'd', 'e', 'f'],
    '20-39': ['g'],
    '40-49': ['h'],
    '50-59': ['i'],
    '>= 60': ['j']
}
CAT_AGE_MAP = dict(itertools.chain(*[
    itertools.product(v,(k,)) for k,v in AGE_MAP.items()
]))


def fetch_data(dept_code):
    # Primer request
    req = perkins.requests.do_request(URL, timeout=60, proxies=proxy)
    soup = BeautifulSoup(req.content, 'html.parser')

    cookies = req.headers['Set-Cookie']
    cookies = [cookie.split(';')[0] for cookie in cookies.split(',')]

    # Form 302: Vigilancia Epidemiologica
    soup = process_request(URL, soup, cookies, {
        **FORM_DATA,
        '__EVENTTARGET': 'ctl00$MainContent$DDL_form',
    })

    # Tabla de mortalidad
    soup = process_request(URL, soup, cookies, {
        **FORM_DATA,
        'ctl00$MainContent$Button1': 'Generar',
    })

    # Agrega/elimina columnas
    for col_name, col_before in FORMAT_COLUMNS:
        soup = process_request(URL, soup, cookies, {
            **FORM_DATA,
            '__CALLBACKID': 'ctl00$MainContent$ASPxPivotGrid1',
            '__CALLBACKPARAM': '|'.join(['c0:D', col_name, col_before, 'true']),
        })

    content = process_request(URL, soup, cookies, {
        **FORM_DATA,
        'ctl00$MainContent$ASPxButton1': '',
    }, raw=True)

    return content


def clean_df(df):
    data_df = df.loc[5:]
    data_df = data_df.loc[:, ~data_df.T.isna().T.all(axis=0)]
    data_df = data_df[~data_df.iloc[:, 2].isna()]

    data_df = data_df.iloc[:, 1:-2]
    data_df = data_df.T.reset_index(drop=True).T.reset_index(drop=True)

    data_df[0] = data_df[0].fillna(method='ffill')
    data_df = data_df.fillna(0)

    return data_df


def format_df(df):
    data_df = pd.DataFrame([])

    for muni, muni_df in df.groupby(0):
        muni_df = muni_df.set_index(1)
        fixed_df = pd.DataFrame([])

        for age, age_df in muni_df.groupby(lambda _: CAT_AGE_MAP[_[0]]):
            fixed_df[age] = age_df.iloc[:, 1:].sum(axis=0).reset_index(drop=True)

        fixed_df.columns = pd.MultiIndex.from_product([[muni], fixed_df.columns])
        data_df = pd.concat([data_df, fixed_df], axis=1)

    data_df.index = pd.MultiIndex.from_product([
        np.arange(len(data_df) / 2, dtype=int) + 1, ['Hombres', 'Mujeres']
    ])
    data_df = data_df.unstack()
    data_df.index = pd.to_datetime(
            data_df.index, unit='W', origin=pd.Timestamp('{}-01-01'.format(year))
    )

    return data_df


if __name__ == '__main__':
    # Busca un proxy que funcione
    if len(sys.argv) > 1 and '--direct' in sys.argv:
        proxy = None
    else:
        proxy = setup_connection(BASE_URL)

    if not proxy:
        print('No available proxy')
        if '--direct' not in sys.argv:
            exit(1)
    else:
        print(proxy)

    df = None
    for dept_name, dept_id in DEPTS.items():
        content = fetch_data(dept_id)

        dept_df = pd.read_excel(io.BytesIO(content), header=None, engine='xlrd')
        dept_df = clean_df(dept_df)
        dept_df = format_df(dept_df)

        dept_df = pd.concat({dept_name: dept_df})
        df = pd.concat([df, dept_df])
