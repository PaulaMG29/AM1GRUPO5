# Calcular el RAAN según la ascensión recta del sol para cualquier fecha y hora
from datetime import datetime, timezone
def calcular_RAAN(epoch_utc: datetime, ltan_h: float) -> float:
    jd = epoch_utc.timestamp() / 86400 + 2440587.5   # fecha juliana 

    d = jd - 2451545.0 

    L = (280.460 + 0.9856474 * d) % 360               # AR del Sol medio 

    return (L + 15 * (ltan_h - 12)) % 360 

print(calcular_RAAN(datetime(2028, 1, 1, 12, 0, 0, tzinfo=timezone.utc), 18))

