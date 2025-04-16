import json

data = {
    "age": 45,
    "smoking": True,
    "yellow_fingers": False,
    "anxiety": True,
    "peer_pressure": False,
    "chronic_disease": True,
    "fatigue": True,
    "allergy": False,
    "wheezing": True,
    "alcohol_consuming": True,
    "coughing": True,
    "shortness_of_breath": True,
    "swallowing_difficulty": False,
    "chest_pain": True
}

json_string = json.dumps(data)

# Membuka file dalam mode tulis ('w')
with open('data.json', 'w') as f:
    # Menulis string JSON ke dalam file
    f.write(json_string)

print("Data JSON berhasil disimpan ke dalam file data.json")