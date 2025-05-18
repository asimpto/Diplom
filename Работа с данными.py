#%% md
# # Выработка энергии
#%%
import pandas as pd
import os

file_path = "data/reestr_sertifikatov_100.xlsx"
excel_data = pd.ExcelFile(file_path)

excluded_sheets = ["Раздел 2 Ген. объекты", "Данные о заключенных СДД","АльтЭнерго - Биогазовая станция","АльтЭнерго - Ветряная Электрост","АльтЭнерго - Солнечная Электрос","Байцуры","Лыковская МГЭС","Мечетлиская микро ГЭС","Норд Гидро - Каллиокоски","Норд Гидро - Ляскеля","Норд Гидро - Рюмякоски","Орловский ГОК","ПМТЭЦ Белый ручей","Фаснальская МГЭС","Фотон - Новокарачаевская МГЭС","Фотон-Учкуланская МГЭС","Цимлянская ГЭС","ЧГК- МГЭС Кокадой","Адыгейская ВЭС 1 этап",'Адыгейская ВЭС 2 этап','Адыгейская ВЭС 3 этап','Азовская ВЭС','ВЭС Тюпкильды','Гуковская ВЭС 1 этап','Гуковская ВЭС 2 этап','Гуковская ВЭС 3 этап','Гуковская ВЭС 4 этап','Гуковская ВЭС 5 этап','Каменская ВЭС 1 этап','Каменская ВЭС 2 этап','Кольская ВЭС','Кочубеевская ВЭС ВЭУ5ВЭУ12','Кочубеевская ВЭС ВЭУ13ВЭУ20','Кочубеевская ВЭС ВЭУ21ВЭУ28','Кочубеевская ВЭС ВЭУ29ВЭУ36','Кочубеевская ВЭС ВЭУ69ВЭУ76','Кочубеевская ВЭС ВЭУ77ВЭУ84','Кочубеевская ВЭС 1-4','Кочубеевская ВЭС 37-44','Кочубеевская ВЭС 45-52','Кочубеевская ВЭС 53-60','Кочубеевская ВЭС 61-68','Марченковская ВЭСВЭУ25-ВЭУ32','Сулинская ВЭС 1 этап','Сулинская ВЭС 2 этап','Фортум ВЭС Ульяновска']

target_sheets = [s for s in excel_data.sheet_names if s not in excluded_sheets]

output_dir = "output/выработка_по_листам"
os.makedirs(output_dir, exist_ok=True)

#%%
# Поиск колонок по маске
def find_column_precise(cols, keyword):
    for col in cols:
        if keyword.lower() in str(col).lower():
            return col
    return None

#%%
import os
import pandas as pd
import numpy as np

MIN_ROWS = 24  # минимальное число строк для сохранения

for sheet in target_sheets:
    try:
        # 1) Чтение листа из Excel
        df = excel_data.parse(sheet, header=3)
        df.columns = df.columns.str.strip()

        # 2) Нахождение нужных колонок
        col_op         = find_column_precise(df.columns, "вид операц")
        col_period     = find_column_precise(df.columns, "расчетный период")
        col_issued     = find_column_precise(df.columns, "выдан сертификат")
        col_unredeemed = find_column_precise(df.columns, "не погашен")
        col_number     = find_column_precise(df.columns, "номер сертификата")

        if not all([col_op, col_period, col_issued, col_unredeemed, col_number]):
            raise ValueError("Не найдены все нужные колонки")

        # 3) Переименование
        df = df.rename(columns={
            col_op:         'Операция',
            col_period:     'Период',
            col_issued:     'Выдано, кВт*ч',
            col_unredeemed: 'Не погашено, кВт*ч',
            col_number:     'Номер'
        })

        # 4) Предобработка
        df['Номер']    = df['Номер'].ffill()
        df['Операция'] = df['Операция'].astype(str).str.lower()

        # 5) Разделение на выпуски и погашения/изменения
        df_vypusk = df[df['Операция'].str.contains('выпуск', na=False)]
        df_izm    = df[df['Операция'].str.contains('изменение', na=False)]
        df_pogash = df[df['Операция'].str.contains('погашение', na=False)]

        df_izm_or_pogash = pd.concat([df_izm, df_pogash]) \
                              .sort_values(['Номер','Операция']) \
                              .drop_duplicates(subset='Номер', keep='first')

        # 6) Слияние по номеру сертификата
        merged = pd.merge(
            df_vypusk[['Номер','Период','Выдано, кВт*ч']],
            df_izm_or_pogash[['Номер','Не погашено, кВт*ч']],
            on='Номер',
            how='left'
        )

        # 7) Конвертация в числа
        merged['Выдано, кВт*ч']       = pd.to_numeric(merged['Выдано, кВт*ч'],    errors='coerce')
        merged['Не погашено, кВт*ч']  = pd.to_numeric(merged['Не погашено, кВт*ч'],errors='coerce')

        # 8) Вычисление выработки и безопасное приведение к nullable Int
        diff = (merged['Выдано, кВт*ч'] - merged['Не погашено, кВт*ч']).round()
        merged['Выработанная энергия, кВт*ч'] = diff.astype('Int64')

        # 9) Финальная таблица
        final_table = (
            merged[['Период','Выработанная энергия, кВт*ч']]
              .dropna(subset=['Выработанная энергия, кВт*ч'])
        )

        # 10) Преобразование периода в Period[M]
        final_table['Период'] = (
            final_table['Период']
              .astype(str)
              .str.extract(r'(\d{2}\.\d{4})')[0]
        )
        final_table['Период'] = pd.to_datetime(
            final_table['Период'], format="%m.%Y", errors='coerce'
        ).dt.to_period("M")

        final_table = final_table.dropna().drop_duplicates()

        # 11) Формирование имени и удаление лидирующих/концевых пробелов
        safe_name = "".join(
            c for c in sheet
            if c.isalnum() or c in (' ','-','_')
        )
        safe_name = safe_name.strip()  # убираем ведущие и конечные пробелы
        csv_path  = os.path.join(output_dir, f"{safe_name}.csv")

        # 12) Сохранение
        final_table.to_csv(csv_path, index=False, encoding='utf-8')

        # 13) Удаление, если строк меньше MIN_ROWS
        row_count = len(final_table)
        if row_count < MIN_ROWS:
            os.remove(csv_path)
            print(f"Файл {csv_path} удалён — всего {row_count} строк (<{MIN_ROWS}).")
        else:
            print(f"Файл {csv_path} сохранён — {row_count} строк.")

    except Exception as e:
        print(f"[{sheet}] Ошибка: {e}")

#%%
import os
import pandas as pd
import numpy as np
import requests
import calendar
from datetime import datetime as dt

# Загрузка координат объектов
coords_df = pd.read_csv(
    'data/Объекты_с_координатами.csv',
    encoding='windows-1251',
    sep=';'
)

# Функция для запроса исторических метео-данных
def fetch_weather(lat, lon, start_date, end_date):
    url = (
        f'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={lat}&longitude={lon}'
        f'&start_date={start_date}&end_date={end_date}'
        f'&daily=temperature_2m_max,temperature_2m_min,'
        f'precipitation_sum,windspeed_10m_max,'
        f'cloudcover_mean,pressure_msl_mean,dewpoint_2m_mean'
        f'&timezone=auto'
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get('daily', {})

# Дневная инсоляция (МДж/м²) по широте и дню года
def solar_insolation(lat_deg, day_of_year):
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365))
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    I_sc = 1367  # Вт/м²
    H0 = (24 * 3600 / np.pi) * I_sc * dr * (
        omega_s * np.sin(phi) * np.sin(delta) +
        np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    return H0 / 1e6  # в МДж/м²

# Суммарная инсоляция за календарный месяц
def monthly_insolation(lat, year, month):
    days = calendar.monthrange(year, month)[1]
    total = 0.0
    for d in range(1, days + 1):
        doy = dt(year, month, d).timetuple().tm_yday
        total += solar_insolation(lat, doy)
    return total

# Список столбцов, которые должны присутствовать в таблице
required_cols = [
    'Температура, °C',
    'Скорость ветра, м/с',
    'Осадки, мм',
    'Облачность, %',
    'Давление, гПа',
    'Точка росы, °C',
    'Инсоляция, МДж/м²'
]

# Обрабатываем каждый объект
for _, coord in coords_df.iterrows():
    obj = coord['Объект']
    lat, lon = coord['Широта'], coord['Долгота']
    path = f"output/выработка_по_листам/{obj}.csv"

    # Пропускаем, если CSV-файла нет
    if not os.path.exists(path):
        print(f"Пропущен {obj}: файл {path} не найден.")
        continue

    df = pd.read_csv(path)

    # 1) Добавляем недостающие колонки
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 2) Находим периоды, где хоть один столбец из required_cols пуст
    mask = df[required_cols].isnull().any(axis=1)
    periods = df.loc[mask, 'Период'].unique()

    # 3) Для каждого такого периода запрашиваем и считаем данные
    for period in periods:
        year, month = map(int, period.split('-'))
        start = f"{year:04d}-{month:02d}-01"
        end_day = calendar.monthrange(year, month)[1]
        end   = f"{year:04d}-{month:02d}-{end_day:02d}"

        daily = fetch_weather(lat, lon, start, end)

        # средние метео-параметры за месяц
        t_max = np.array(daily.get('temperature_2m_max', []))
        t_min = np.array(daily.get('temperature_2m_min', []))
        wind  = np.array(daily.get('windspeed_10m_max', []))
        prec  = np.array(daily.get('precipitation_sum', []))
        cloud = np.array(daily.get('cloudcover_mean', []))
        press = np.array(daily.get('pressure_msl_mean', []))
        dew   = np.array(daily.get('dewpoint_2m_mean', []))

        ins = monthly_insolation(lat, year, month)

        # записываем в DataFrame
        idx = df[df['Период'] == period].index
        df.loc[idx, 'Температура, °C']     = float(((t_max + t_min) / 2).mean())
        df.loc[idx, 'Скорость ветра, м/с'] = float(wind.mean())
        df.loc[idx, 'Осадки, мм']          = float(prec.mean())
        df.loc[idx, 'Облачность, %']       = float(cloud.mean())
        df.loc[idx, 'Давление, гПа']       = float(press.mean())
        df.loc[idx, 'Точка росы, °C']      = float(dew.mean())
        df.loc[idx, 'Инсоляция, МДж/м²']    = ins

    # 4) Сохраняем только если были пропуски
    if len(periods) > 0:
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"Заполнено и сохранено для {obj}: {len(periods)} период(а/ов)")
    else:
        print(f"Нет пропусков для {obj}, пропущено.")

#%%
# Путь к директории с таблицами
folder_path = 'output/выработка_по_листам'

weather_columns = [
    'Температура, °C',
    'Скорость ветра, м/с',
    'Осадки, мм',
    'Облачность, %',
    'Давление, гПа',
    'Точка росы, °C',
    'Инсоляция, МДж/м²'
]

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)

        # Проверяем, что все нужные столбцы есть
        present_cols = [col for col in weather_columns if col in df.columns]
        if present_cols:
            df[present_cols] = df[present_cols].round(2)

            # Новое имя файла с расширением .xlsx
            excel_path = filepath.replace('.csv', '.xlsx')
            excel_path = excel_path.replace('выработка_по_листам', 'готовая_выработка_excel')

            df.to_excel(excel_path, index=False)
            print(f"Округлено и сохранено в Excel: {excel_path}")
        else:
            print(f"Пропущен (нет погодных данных): {filename}")

#%%
