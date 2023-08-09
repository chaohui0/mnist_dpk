def rescale(data, **kwargs):
    """重缩放数据到0~1范围内

    Args:
        ##TODO

    Returns:
        ##TODO

    """
    max_value = kwargs.get('max_value', 255)
    min_value = kwargs.get('min_value', 0)
    data = (data-min_value)/(max_value-min_value)

    return data

