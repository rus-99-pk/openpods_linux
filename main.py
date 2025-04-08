# -*- coding: utf-8 -*-
"""
Скрипт для мониторинга уровня заряда Apple AirPods через Bluetooth Low Energy (BLE) advertisements.
Script to monitor Apple AirPods battery levels via Bluetooth Low Energy (BLE) advertisements.
"""

# Импорт необходимых библиотек
# Import necessary libraries
from argparse import ArgumentParser # Для получения аргументов командной строки / For getting command line arguments
from bleak import BleakScanner  # Для сканирования BLE устройств / For scanning BLE devices
from asyncio import new_event_loop, set_event_loop, get_event_loop, sleep as asyncio_sleep # Для асинхронной работы / For asynchronous operations
from time import sleep as time_sleep, time_ns # Для пауз и временных меток / For pauses and timestamps
from binascii import hexlify # Для конвертации байтов в hex / For converting bytes to hex
from json import dumps # Для форматирования вывода в JSON / For formatting output to JSON
from datetime import datetime # Для добавления временной метки к данным / For adding timestamps to data
from os import system # Потенциально для системных команд (не используется в текущей версии) / Potentially for system commands (not used in current version)
import sys

# --- Конфигурация / Configuration ---

# Длительность обновления (в секундах): как часто сканировать и обновлять данные
# Update duration (in seconds): how often to scan and update data
UPDATE_DURATION = 2

# Минимальный уровень сигнала (RSSI) для обработки данных устройства. Более высокое значение (ближе к 0) означает более сильный сигнал.
# Minimum Received Signal Strength Indicator (RSSI) to process device data. Higher value (closer to 0) means stronger signal.
MIN_RSSI = -60

# Идентификатор производителя Apple в данных BLE
# Apple's manufacturer ID in BLE data
AIRPODS_MANUFACTURER = 76

# Ожидаемая длина данных производителя AirPods в hex-символах (27 байт * 2 символа/байт)
# Expected length of AirPods manufacturer data in hex characters (27 bytes * 2 chars/byte)
AIRPODS_DATA_LENGTH = 54

# Максимальное время (в наносекундах), в течение которого маячок считается "недавним" для выбора лучшего сигнала
# Maximum time (in nanoseconds) a beacon is considered "recent" for selecting the best signal
RECENT_BEACONS_MAX_T_NS = 10 * 1_000_000_000  # 10 секунд / 10 Seconds

# --- Глобальные переменные / Global variables ---

# Список для хранения информации о недавно обнаруженных маячках (время, устройство, RSSI)
# List to store information about recently detected beacons (time, device, RSSI)
recent_beacons = []

# --- Функции / Functions ---

def format_for_panel(data):
    """
    Форматирует данные AirPods в строку для панели.
    Formats AirPods data into a string for the panel.
    """
    if not data or data.get("status") != 1:
        return "AirPods: N/A" # Или иконку ошибки / Or an error icon

    charge = data['charge']
    left = charge['left']
    right = charge['right']
    case = charge['case']

    charging_left = data['charging_left']
    charging_right = data['charging_right']
    charging_case = data['charging_case']

    # Формируем строки для каждого элемента с иконкой зарядки ⚡
    # Format strings for each element with charging icon ⚡
    # Используем "?" если заряд -1 (неизвестен) / Use "?" if charge is -1 (unknown)
    left_str = f"{left}%" if left != -1 else "?"
    if charging_left and left != -1:
        left_str += "⚡"

    right_str = f"{right}%" if right != -1 else "?"
    if charging_right and right != -1:
        right_str += "⚡"

    case_str = f"{case}%" if case != -1 else "?"
    if charging_case and case != -1:
        case_str += "⚡"

    # Собираем финальную строку.
    # Assemble the final string.
    return f"L {left_str} | R {right_str} | C {case_str}"

def get_best_result(device, advertisement_data):
    """
    Отслеживает недавние BLE-маячки от одного и того же устройства и возвращает
    объект устройства, соответствующий маячку с самым сильным сигналом (RSSI) за последнее время.
    Это помогает получить наиболее стабильные данные, если принимается несколько пакетов.

    Tracks recent BLE beacons from the same device and returns the device object
    corresponding to the beacon with the strongest signal (RSSI) recently.
    This helps in getting more stable data if multiple packets are received.

    Args:
        device (BLEDevice): Объект обнаруженного устройства / The detected device object.
        advertisement_data (AdvertisementData): Данные рекламы устройства / The advertisement data of the device.

    Returns:
        BLEDevice: Объект устройства с самым сильным сигналом за последнее время / The device object with the strongest signal recently.
    """
    current_time = time_ns()

    # Добавляем текущее обнаруженное устройство и его RSSI в список недавних
    # Add the currently detected device and its RSSI to the list of recent beacons
    recent_beacons.append({
        "time": current_time,
        "device": device,
        "rssi": advertisement_data.rssi
    })

    strongest_beacon = None  # Хранит запись о маячке с самым сильным сигналом / Stores the record of the strongest beacon
    i = 0
    # Итерация по списку недавних маячков для очистки старых и поиска самого сильного
    # Iterate through the list of recent beacons to clean up old ones and find the strongest
    while i < len(recent_beacons):
        # Удаляем старые записи (старше RECENT_BEACONS_MAX_T_NS)
        # Remove old entries (older than RECENT_BEACONS_MAX_T_NS)
        if current_time - recent_beacons[i]["time"] > RECENT_BEACONS_MAX_T_NS:
            recent_beacons.pop(i)
            continue # Переходим к следующей итерации без инкремента i, т.к. список изменился / Continue to the next iteration without incrementing i, as the list changed

        # Ищем маячок с самым высоким RSSI (ближе к 0)
        # Find the beacon with the highest RSSI (closer to 0)
        if (strongest_beacon is None or strongest_beacon["rssi"] < recent_beacons[i]["rssi"]):
            strongest_beacon = recent_beacons[i]
        i += 1

    # Если самый сильный маячок найден и его MAC-адрес совпадает с текущим обнаруженным устройством,
    # обновляем объект 'device' в записи strongest_beacon на самый свежий.
    # If the strongest beacon is found and its MAC address matches the currently detected device,
    # update the 'device' object in the strongest_beacon record to the most recent one.
    if (strongest_beacon is not None and
        strongest_beacon["device"].address == device.address):
        # Это может быть полезно, если объект device содержит обновленную информацию, хотя обычно адреса достаточно.
        # This might be useful if the device object contains updated info, though usually the address is sufficient.
        strongest_beacon["device"] = device

    # Возвращаем объект BLEDevice из записи о самом сильном маячке, если он найден,
    # иначе возвращаем текущее устройство (если это был единственный маячок в недавнем списке).
    # Return the BLEDevice object from the strongest beacon record if found,
    # otherwise return the current device (if it was the only beacon in the recent list).
    return strongest_beacon["device"] if strongest_beacon else device

async def get_device():
    """
    Асинхронно сканирует эфир в поисках BLE-устройств, фильтрует рекламу AirPods
    и возвращает данные производителя в hex-формате от устройства с лучшим сигналом.

    Asynchronously scans for BLE devices, filters for AirPods advertisements,
    and returns the manufacturer data in hex format from the device with the best signal.

    Returns:
        bytes or False: Hex-строка данных производителя (в виде байтов) или False, если данные не найдены. /
                        Hex string of manufacturer data (as bytes) or False if no data found.
    """
    device_data = None # Переменная для хранения найденных данных / Variable to store found data

    # Функция обратного вызова, вызывается для каждого обнаруженного BLE-устройства
    # Callback function, invoked for each discovered BLE device
    def detection_callback(device, advertisement_data):
        nonlocal device_data # Позволяет изменять device_data во внешней области видимости / Allows modification of device_data in the outer scope

        # Получаем устройство с лучшим сигналом за последнее время
        # Get the device with the best signal recently
        best_device = get_best_result(device, advertisement_data)

        # Проверяем, соответствует ли устройство критериям: достаточный RSSI и ID производителя Apple
        # Check if the device meets the criteria: sufficient RSSI and Apple manufacturer ID
        if (advertisement_data.rssi >= MIN_RSSI and
            AIRPODS_MANUFACTURER in advertisement_data.manufacturer_data):

            # Получаем данные производителя как массив байт
            # Get the manufacturer data as a byte array
            manufacturer_bytes = advertisement_data.manufacturer_data[AIRPODS_MANUFACTURER]
            # Конвертируем в hex-строку (в виде байтов)
            # Convert to a hex string (as bytes)
            data_hex = hexlify(manufacturer_bytes)
            # Проверяем длину данных (должна быть AIRPODS_DATA_LENGTH hex-символов)
            # Check the data length (should be AIRPODS_DATA_LENGTH hex characters)
            if len(data_hex) == AIRPODS_DATA_LENGTH:
                # Сохраняем данные, если все условия выполнены
                # Save the data if all conditions are met
                # Примечание: Если найдено несколько подходящих устройств за время сканирования,
                # будет сохранено последнее. Логика get_best_result помогает, но не гарантирует
                # выбор данных именно от устройства с лучшим сигналом *на момент сохранения*.
                # Note: If multiple suitable devices are found during the scan time,
                # the last one will be saved. The get_best_result logic helps but doesn't guarantee
                # selecting data from the device with the best signal *at the moment of saving*.
                device_data = data_hex

    # Создаем и запускаем сканер BLE
    # Create and start the BLE scanner
    scanner = BleakScanner(detection_callback=detection_callback)
    await scanner.start()
    # Ожидаем некоторое время для сбора данных (5 секунд)
    # Wait for some time to collect data (5 seconds)
    await asyncio_sleep(5)
    # Останавливаем сканер
    # Stop the scanner
    await scanner.stop()

    # Возвращаем найденные данные или False
    # Return the found data or False
    return device_data if device_data else False

def get_data_hex():
    """
    Синхронная обертка для асинхронной функции get_device().
    Запускает новый цикл событий asyncio для выполнения сканирования.

    Synchronous wrapper for the asynchronous get_device() function.
    Runs a new asyncio event loop to perform the scan.

    Returns:
        bytes or False: Результат выполнения get_device() / The result of get_device().
    """
    # Создаем новый цикл событий asyncio / Create a new asyncio event loop
    new_loop = new_event_loop()
    # Устанавливаем его как текущий для этого потока / Set it as the current loop for this thread
    set_event_loop(new_loop)
    # Получаем текущий цикл событий / Get the current event loop
    loop = get_event_loop()
    # Запускаем асинхронную функцию get_device() и ждем ее завершения
    # Run the asynchronous function get_device() and wait for its completion
    result = loop.run_until_complete(get_device())
    # Закрываем цикл событий / Close the event loop
    loop.close()
    # Возвращаем результат / Return the result
    return result

def get_data():
    """
    Получает сырые hex-данные от AirPods, парсит их и возвращает словарь (JSON-совместимый)
    с информацией о заряде, статусе зарядки, модели и т.д.

    Retrieves raw hex data from AirPods, parses it, and returns a dictionary (JSON-compatible)
    with information about battery levels, charging status, model, etc.

    Returns:
        dict: Словарь с данными AirPods или словарь с сообщением об ошибке. /
              Dictionary with AirPods data or a dictionary with an error message.
    """
    # Получаем сырые hex-данные (в виде байтов)
    # Get the raw hex data (as bytes)
    raw = get_data_hex()

    # Если данные не получены (сканирование не нашло AirPods или сигнал слишком слабый)
    # If no data was received (scan didn't find AirPods or signal was too weak)
    if not raw:
        return dict(status=0, message="AirPods advertisement not found or signal too weak.")

    try:
        # Декодируем байтовую hex-строку в обычную строку UTF-8 для удобства работы
        # Decode the byte hex string into a regular UTF-8 string for easier handling
        raw_str = raw.decode("utf-8")

        # Определяем, перепутаны ли местами данные левого и правого наушников
        # Determine if the left and right earbud data are swapped
        flip = is_flipped(raw) # Передаем оригинальные байты, так как is_flipped ожидает их / Pass original bytes as is_flipped expects them

        # --- Определение модели / Model Identification ---
        # Используем 8-й символ (индекс 7) hex-строки для определения модели
        # Use the 8th character (index 7) of the hex string to determine the model
        model_char = raw_str[7]
        if model_char == 'e': # AirPods Pro (1st/2nd gen often use 'e')
            model = "AirPodsPro"
        elif model_char == '3': # AirPods 3rd gen
            model = "AirPods3"
        elif model_char == 'f': # AirPods 2nd gen
             model = "AirPods2"
        elif model_char == '2': # AirPods 1st gen
            model = "AirPods1"
        elif model_char == 'a': # AirPods Max
            model = "AirPodsMax"
        else:
            # Неизвестная или новая модель / Unknown or new model
            model = f"unknown ({model_char})"

        # --- Расчет уровня заряда / Battery Level Calculation ---
        # Заряд представлен младшими 4 битами (nybble) соответствующего байта (hex-символа).
        # Значения 0-9 соответствуют 0-90% с шагом 10%.
        # Значение 10 (0xA) соответствует 100%.
        # Значение 15 (0xF) означает "неизвестно" или "не подключено".
        # Battery level is represented by the lower 4 bits (nybble) of the corresponding byte (hex character).
        # Values 0-9 correspond to 0-90% in 10% steps.
        # Value 10 (0xA) corresponds to 100%.
        # Value 15 (0xF) means "unknown" or "disconnected".

        # Левый наушник / Left Earbud
        # Индекс символа зависит от флага 'flip' / Character index depends on the 'flip' flag
        left_hex_char = raw_str[12 if flip else 13]
        # Преобразуем hex-символ в целое число (0-15) / Convert hex char to integer (0-15)
        left_status_val = int(left_hex_char, 16) # Не нужно & 0x0F, т.к. int() и так дает 0-15 / No need for & 0x0F, int() already gives 0-15
        if left_status_val == 0xA: # 10 decimal -> 100%
            left_status = 100
        elif 0 <= left_status_val <= 9: # 0-9 decimal -> 0-90%
            left_status = left_status_val * 10
        else: # 0xF (15) or other unexpected values -> Unknown/Disconnected
            left_status = -1 # Используем -1 для обозначения "неизвестно" / Use -1 to indicate "unknown"

        # Правый наушник / Right Earbud
        # Индекс символа зависит от флага 'flip' / Character index depends on the 'flip' flag
        right_hex_char = raw_str[13 if flip else 12]
        right_status_val = int(right_hex_char, 16)
        if right_status_val == 0xA: # 100%
            right_status = 100
        elif 0 <= right_status_val <= 9: # 0-90%
            right_status = right_status_val * 10
        else: # Unknown/Disconnected
            right_status = -1

        # Кейс / Case
        # Индекс символа для кейса фиксирован / Character index for the case is fixed
        case_hex_char = raw_str[15]
        case_status_val = int(case_hex_char, 16)
        if case_status_val == 0xA: # 100%
            case_status = 100
        elif 0 <= case_status_val <= 9: # 0-90%
            case_status = case_status_val * 10
        else: # Unknown/Disconnected
            case_status = -1

        # --- Статус зарядки / Charging Status ---
        # Байт по индексу 14 содержит флаги зарядки для левого, правого и кейса
        # The byte at index 14 contains charging flags for left, right, and case
        charging_status_byte = int(raw_str[14], 16)
        # Определяем битовые маски для левого и правого в зависимости от 'flip'
        # Determine bitmasks for left and right depending on 'flip'
        # Маска 0b00000001 (1): Правый (если не flip) / Левый (если flip)
        # Mask 0b00000001 (1): Right (if not flip) / Left (if flip)
        # Маска 0b00000010 (2): Левый (если не flip) / Правый (если flip)
        # Mask 0b00000010 (2): Left (if not flip) / Right (if flip)
        left_charging_mask = 0b00000010 if flip else 0b00000001
        right_charging_mask = 0b00000001 if flip else 0b00000010
        # Маска 0b00000100 (4): Кейс
        # Mask 0b00000100 (4): Case
        case_charging_mask = 0b00000100

        # Проверяем установлены ли соответствующие биты с помощью побитового И (&)
        # Check if the corresponding bits are set using bitwise AND (&)
        charging_left = (charging_status_byte & left_charging_mask) != 0
        charging_right = (charging_status_byte & right_charging_mask) != 0
        charging_case = (charging_status_byte & case_charging_mask) != 0

        # Формируем результат в виде словаря
        # Format the result as a dictionary
        return dict(
            status=1, # Статус 1 означает успешное получение и парсинг данных / Status 1 means data successfully received and parsed
            charge=dict(
                left=left_status,  # Заряд левого (%zie (% или -1) / Left charge (% or -1)
                right=right_status, # Заряд правого (% или -1) / Right charge (% or -1)
                case=case_status   # Заряд кейса (% или -1) / Case charge (% or -1)
            ),
            charging_left=charging_left,   # Заряжается ли левый (True/False) / Is left charging (True/False)
            charging_right=charging_right, # Заряжается ли правый (True/False) / Is right charging (True/False)
            charging_case=charging_case,   # Заряжается ли кейс (True/False) / Is case charging (True/False)
            model=model,                  # Обнаруженная модель / Detected model
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # Текущая дата и время / Current date and time
            raw=raw_str # Возвращаем декодированную hex-строку для отладки / Return the decoded hex string for debugging
        )

    except (IndexError, ValueError, TypeError) as e:
        # Перехватываем возможные ошибки при обработке данных (неправильная длина, не hex символы и т.д.)
        # Catch potential errors during data processing (incorrect length, non-hex characters, etc.)
        return dict(
            status=0, # Статус 0 означает ошибку / Status 0 means error
            message=f"Error processing AirPods data: {e}", # Сообщение об ошибке / Error message
            raw=raw.decode("utf-8", errors='ignore') if raw else None # Показываем сырые данные (если есть) для отладки / Show raw data (if available) for debugging
        )

def is_flipped(raw):
    """
    Определяет, перепутаны ли местами данные левого и правого наушников в данных BLE.
    Это определяется битом 1 (второй справа) байта по индексу 10.

    Determines if the left and right earbud data are swapped in the BLE data.
    This is determined by bit 1 (second from the right) of the byte at index 10.

    Args:
        raw (bytes): Сырые hex-данные в виде байтов / Raw hex data as bytes.

    Returns:
        bool: True, если данные перепутаны (flipped), False в противном случае или при ошибке. /
              True if the data is flipped, False otherwise or on error.
    """
    # Проверка на None и достаточную длину для безопасности
    # Check for None and sufficient length for safety
    # Индекс 10 требует как минимум 11 байт (22 hex символа)
    # Index 10 requires at least 11 bytes (22 hex characters)
    if raw is None or len(raw) < 22: # Работаем с hex-строкой, нужно 2 символа на байт / Working with hex string, need 2 chars per byte
         # Если данных недостаточно, считаем, что не перепутано (или можно вызвать ошибку)
         # If data is insufficient, assume not flipped (or could raise an error)
         return False
    try:
        # Получаем символ hex по индексу 10 (11-й символ)
        # Get the hex character at index 10 (11th character)
        # Мы передаем байты `raw` из get_data, поэтому используем chr() для получения символа
        # We pass bytes `raw` from get_data, so use chr() to get the character
        flip_byte_char = chr(raw[10])
        # Конвертируем hex-символ в число (0-15)
        # Convert hex character to number (0-15)
        flip_val = int(flip_byte_char, 16)
        # Проверяем бит 1: (flip_val & 0b0010) == 0
        # Check bit 1: (flip_val & 0b0010) == 0
        # Если бит 1 равен 0, то данные перепутаны (flipped = True)
        # If bit 1 is 0, the data is flipped (flipped = True)
        return (flip_val & 0x02) == 0
    except (ValueError, IndexError):
        # В случае ошибки при разборе флага (например, не hex символ), считаем False
        # In case of error parsing the flag (e.g., non-hex character), assume False
        return False

def run(output_format='json', output_file=None): # Добавляем аргументы / Add arguments
    """
    Основной цикл программы. Периодически сканирует AirPods, получает данные,
    форматирует их и выводит в нужном формате.

    Main program loop. Periodically scans for AirPods, retrieves data,
    formats it, and outputs it in the desired format.

    Args:
        output_format (str): Формат вывода ('json' или 'text') / Output format ('json' or 'text')
        output_file (str, optional): Путь к файлу для записи JSON / Path to file for JSON output
    """
    # Бесконечный цикл, если не используется формат 'text' (т.к. индикатор сам перезапускает)
    # Infinite loop unless using 'text' format (as the indicator restarts it)
    # Для текстового режима достаточно одного запуска / For text mode, one run is enough
    is_single_run = (output_format == 'text')

    while True:
        data = get_data()

        if output_format == 'text':
            # Выводим отформатированный текст и выходим (индикатор перезапустит)
            # Print formatted text and exit (indicator will restart)
            print(format_for_panel(data))
            break # Выход из цикла / Exit loop

        # --- Старая логика вывода JSON ---
        # --- Old JSON output logic ---
        elif data and data.get("status") == 1:
            json_data = dumps(data, indent=4, ensure_ascii=False)
            if output_file:
                try:
                    with open(output_file, "a", encoding='utf-8') as f:
                        f.write(json_data + "\n")
                except IOError as e:
                    print(f"Error writing to file {output_file}: {e}")
                    print("Outputting to console instead:")
                    print(json_data)
                    output_file = None
            else:
                print(json_data)
        elif data:
            error_msg = data.get('message', 'Unknown error occurred.')
            raw_info = data.get('raw')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {error_msg}" + (f" Raw: {raw_info}" if raw_info else ""), file=sys.stderr) # Вывод ошибок в stderr / Print errors to stderr
        # --- Конец старой логики ---
        # --- End of old logic ---

        if is_single_run: # Если уже выполнили один раз для text, выходим
             break
        else: # Для JSON режима делаем паузу
            time_sleep(UPDATE_DURATION)

if __name__ == '__main__':
    # Парсер аргументов командной строки / Command line argument parser
    parser = ArgumentParser(description="AirPods Battery Monitor")
    parser.add_argument(
        "--format",
        choices=['json', 'text'],
        default='json', # По умолчанию JSON / Default is JSON
        help="Output format: json (default) or text (for panel indicators)"
    )
    parser.add_argument(
        "-o", "--output-file",
        metavar="FILE",
        help="Append JSON output to FILE (ignored if format is text)"
    )
    args = parser.parse_args()

    # Определяем файл вывода только для JSON режима
    # Determine output file only for JSON mode
    output_file_arg = args.output_file if args.format == 'json' else None

    try:
        if args.format == 'json':
            print("Starting AirPods monitor (JSON mode)... Press Ctrl+C to exit.", file=sys.stderr)
        # Запускаем run с переданными аргументами / Run 'run' with parsed arguments
        run(output_format=args.format, output_file=output_file_arg)
    except KeyboardInterrupt:
        if args.format == 'json':
            print("\nExiting script...", file=sys.stderr)
    finally:
        if args.format == 'json':
            print("Monitor stopped.", file=sys.stderr)
        pass # Пустой блок для завершения / Empty block for completion