import math


def bearing_frequencies(n, d_p, d_b, phi, rpm):
    """
    Calculate bearing characteristic frequencies.

    :param n: 滚珠个数 Number of rolling elements
    :param d_p: 轴承滚道节径(mm) Pitch diameter
    :param d_b: 滚珠直径(mm) Rolling element diameter
    :param phi: 轴承接触角Contact angle (degrees)
    :param rpm: 内圈转速Rotational speed (revolutions per minute)
    :return: A tuple containing (FTF, BPFI, BPFO, BSF)
    """
    # Convert contact angle to radians
    phi_rad = math.radians(phi)

    # Calculate the fundamental train frequency (FTF)
    FTF = 0.5 * rpm / 60 * (1 - d_b / d_p * math.cos(phi_rad))

    # Calculate the ball pass frequency of the inner race (BPFI)
    BPFI = 0.5 * n * rpm / 60 * (1 + d_b / d_p * math.cos(phi_rad))

    # Calculate the ball pass frequency of the outer race (BPFO)
    BPFO = 0.5 * n * rpm / 60 * (1 - d_b / d_p * math.cos(phi_rad))

    # Calculate the ball spin frequency (BSF)
    BSF = 0.5 * d_p / d_b * rpm / 60 * (1 - (d_b / d_p * math.cos(phi_rad)) ** 2)

    return FTF, BPFI, BPFO, BSF


# Bearing parameters
n = 9  # 滚珠个数 Number of rolling elements
d_p = 39.0398  # 轴承滚道节径(mm) Pitch diameter
d_b = 7.94004  # 滚珠直径(mm) Rolling element diameter
phi = 0  # 轴承接触角Contact angle (degrees)
rpm = 1730  # 内圈转速Rotational speed (revolutions per minute)

# Calculate bearing characteristic frequencies
FTF, BPFI, BPFO, BSF = bearing_frequencies(n, d_p, d_b, phi, rpm)

print("保持架频率 Fundamental Train Frequency (FTF):", FTF)
print("滚动体通过内圈频率 Ball Pass Frequency of Inner Race (BPFI):", BPFI)
print("滚动体通过外圈频率 Ball Pass Frequency of Outer Race (BPFO):", BPFO)
print("滚动体自转频率  Ball Spin Frequency (BSF):", BSF)
