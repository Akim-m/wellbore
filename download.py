import kagglehub

# Download latest version
path = kagglehub.competition_download("rogii-wellbore-geology-prediction")

print("Path to competition files:", path)
