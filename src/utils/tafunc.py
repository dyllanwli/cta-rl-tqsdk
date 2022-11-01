from datetime import datetime

def _datetime_to_timestamp_nano(dt: datetime) -> int:
    # timestamp() 返回值精度为 microsecond，直接乘以 1e9 可能有精度问题
    return int(dt.timestamp() * 1000000) * 1000

def _str_to_timestamp_nano(current_datetime: str, fmt="%Y-%m-%d %H:%M:%S.%f") -> int:
    return _datetime_to_timestamp_nano(datetime.strptime(current_datetime, fmt))

def _to_ns_timestamp(input_time):
    if type(input_time) in {int, float, np.float64, np.float32, np.int64, np.int32}:  # 时间戳
        if input_time > 2 ** 32:  # 纳秒( 将 > 2*32数值归为纳秒级)
            return int(input_time)
        else:  # 秒
            return int(input_time * 1e9)
    elif isinstance(input_time, str):  # str 类型时间
        return _str_to_timestamp_nano(input_time)
    elif isinstance(input_time, datetime):  # datetime 类型时间
        return _datetime_to_timestamp_nano(input_time)
    else:
        raise TypeError("暂不支持此类型的转换")

def time_to_s_timestamp(input_time):
    return int(_to_ns_timestamp(input_time) / 1e9)