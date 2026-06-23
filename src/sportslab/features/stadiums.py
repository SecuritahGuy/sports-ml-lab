"""NFL stadium coordinates for weather lookup via meteostat."""

STADIUM_COORDS: dict[str, tuple[float, float, str]] = {
    "ATL97": (33.7550, -84.4000, "America/New_York"),  # Mercedes-Benz Stadium
    "BAL00": (39.2780, -76.6227, "America/New_York"),  # M&T Bank Stadium
    "BOS00": (42.0909, -71.2643, "America/New_York"),  # Gillette Stadium
    "BUF00": (42.7738, -78.7869, "America/New_York"),  # Highmark Stadium (New Era Field)
    "CAR00": (35.2258, -80.8528, "America/New_York"),  # Bank of America Stadium
    "CHI98": (41.8623, -87.6167, "America/Chicago"),  # Soldier Field
    "CIN00": (39.0954, -84.5166, "America/New_York"),  # Paycor Stadium
    "CLE00": (41.5061, -81.6995, "America/New_York"),  # Huntington Bank Field (FirstEnergy)
    "DAL00": (32.7475, -97.0928, "America/Chicago"),  # AT&T Stadium
    "DEN00": (39.7439, -105.0201, "America/Denver"),  # Empower Field at Mile High
    "DET00": (42.3400, -83.0456, "America/Detroit"),  # Ford Field
    "FRA00": (50.0686, 8.6455, "Europe/Berlin"),  # Deutsche Bank Park
    "GER00": (48.2188, 11.6247, "Europe/Berlin"),  # Allianz Arena
    "GNB00": (44.5013, -88.0622, "America/Chicago"),  # Lambeau Field
    "HOU00": (29.6847, -95.4108, "America/Chicago"),  # NRG Stadium
    "IND00": (39.7601, -86.1639, "America/Indianapolis"),  # Lucas Oil Stadium
    "JAX00": (30.3240, -81.6373, "America/New_York"),  # TIAA Bank Field
    "KAN00": (39.0489, -94.4839, "America/Chicago"),  # Arrowhead Stadium
    "LAX01": (33.8535, -118.3391, "America/Los_Angeles"),  # SoFi Stadium
    "LON00": (51.5560, -0.2795, "Europe/London"),  # Wembley Stadium
    "LON02": (51.6034, -0.0657, "Europe/London"),  # Tottenham Hotspur Stadium
    "MEX00": (19.3030, -99.1504, "America/Mexico_City"),  # Estadio Azteca
    "MIA00": (25.9580, -80.2389, "America/New_York"),  # Hard Rock Stadium
    "MIN01": (44.9740, -93.2577, "America/Chicago"),  # U.S. Bank Stadium
    "NAS00": (36.1664, -86.7714, "America/Chicago"),  # Nissan Stadium
    "NOR00": (29.9509, -90.0811, "America/Chicago"),  # Caesars Superdome
    "NYC01": (40.8135, -74.0743, "America/New_York"),  # MetLife Stadium
    "PHI00": (39.9008, -75.1675, "America/New_York"),  # Lincoln Financial Field
    "PHO00": (33.5276, -112.2626, "America/Phoenix"),  # State Farm Stadium
    "PIT00": (40.4468, -80.0158, "America/New_York"),  # Acrisure Stadium (Heinz Field)
    "SAO00": (-23.5455, -46.4743, "America/Sao_Paulo"),  # Arena Corinthians
    "SEA00": (47.5951, -122.3316, "America/Los_Angeles"),  # Lumen Field
    "SFO01": (37.4030, -121.9694, "America/Los_Angeles"),  # Levi's Stadium
    "TAM00": (27.9759, -82.5033, "America/New_York"),  # Raymond James Stadium
    "VEG00": (36.0907, -115.1838, "America/Los_Angeles"),  # Allegiant Stadium
    "WAS00": (38.8647, -76.8855, "America/New_York"),  # FedExField
}
