import pandas as pd


def prepare_features(df, artifacts):
    df = change_types(df)
    df = fill_nulls(df, artifacts)
    df = fix_screen_resolution(df, artifacts)
    df = apply_top_categories(df, artifacts)
    df = fix_outliers(df, artifacts)
    df = feature_engineering(df, artifacts)
    return df


def change_types(df):
    df_copy = df.copy()
    df_copy['visit_date'] = pd.to_datetime(df_copy['visit_date'])
    df_copy['visit_time'] = pd.to_datetime(df_copy['visit_time'], format='%H:%M:%S').dt.time

    return df_copy


def fill_nulls(df, artifacts):
    df_copy = df.copy()
    df_copy['utm_keyword'] = df_copy['utm_keyword'].fillna('not_set')
    df_copy['utm_campaign'] = df_copy['utm_campaign'].fillna('not_set')
    df_copy['utm_adcontent'] = df_copy['utm_adcontent'].fillna('not_set')
    df_copy['device_brand'] = df_copy['device_brand'].fillna('not_set')
    df_copy['utm_source'] = df_copy['utm_source'].fillna('not_set')
    df_copy = df_copy.drop(columns=['device_model'])

    os_map = artifacts['os_map']
    df_copy['device_os'] = df_copy.apply(
        lambda row: os_map.get((row['device_category'], row['device_brand']), 'not_set')
        if pd.isna(row['device_os']) else row['device_os'], axis=1
    )

    return df_copy


def fix_outliers(df, artifacts):
    df_copy = df.copy()

    boundaries = artifacts['boundaries']

    for col, (lower_bound, upper_bound) in boundaries.items():
        df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

    return df_copy


def feature_engineering(df, artifacts):
    df_copy = df.copy()
    df_copy['visit_month'] = df_copy['visit_date'].dt.month
    df_copy['visit_day'] = df_copy['visit_date'].dt.day
    df_copy['visit_dayofweek'] = df_copy['visit_date'].dt.dayofweek
    df_copy['visit_hour'] = pd.to_datetime(df_copy['visit_time'], format='%H:%M:%S').dt.hour

    df_copy['is_russia'] = (df_copy['geo_country'] == 'Russia').astype(int)

    first_visit_map = artifacts['first_visit_map']
    df_copy['is_new_client'] = df_copy['client_id'].map(
        lambda x: 0 if x in first_visit_map.index else 1
    )

    df_copy['city_group'] = df_copy['geo_city'].apply(
        lambda x: 'moscow' if 'Moscow' in str(x)
        else 'spb' if 'Saint Petersburg' in str(x)
        else 'other'
    )

    organic = ['organic', 'referral', '(none)']
    df_copy['is_organic'] = df_copy['utm_medium'].isin(organic).astype(int)

    social_sources = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                      'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                      'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df_copy['is_social'] = df_copy['utm_source'].isin(social_sources).astype(int)

    df_copy = df_copy.drop(columns=['session_id', 'client_id', 'visit_date', 'visit_time', 'geo_country', 'geo_city'])

    return df_copy


def apply_top_categories(df, artifacts):
    df_copy = df.copy()

    top_categories = artifacts['top_categories']

    for col, top_values in top_categories.items():
        df_copy[col] = df_copy[col].where(
            df_copy[col].isin(top_values), other='other'
        )

    return df_copy


def fix_screen_resolution(df, artifacts):
    df_copy = df.copy()
    df_copy['device_screen_resolution'] = df_copy['device_screen_resolution'].replace('(not set)', '0x0')
    df_copy['screen_width'] = df_copy['device_screen_resolution'].str.split('x').str[0].astype(int)
    df_copy['screen_height'] = df_copy['device_screen_resolution'].str.split('x').str[1].astype(int)
    df_copy = df_copy.drop(columns=['device_screen_resolution'])

    for col in ['screen_width', 'screen_height']:
        df_copy[col] = df_copy[col].replace(0, None)
        mode_map = artifacts['screen_mode_map'][col]
        df_copy[col] = df_copy.apply(
            lambda row: mode_map[row['device_category']]
            if pd.isna(row[col]) else row[col], axis=1
        )
    return df_copy
