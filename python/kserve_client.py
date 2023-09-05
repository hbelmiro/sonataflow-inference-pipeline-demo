import os
import requests

def send_request_with_payload(url, payload_file, response_path):
    # Read the payload from the file
    with open(payload_file, 'rb') as file:
        payload_data = file.read()

    # Send the POST request with the payload
    response = requests.post(url, data=payload_data)

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        response_file = 'kserve_response.json'
        # Save the response payload to the specified file
        with open(response_path + response_file, 'wb') as file:
            file.write(response.content)
        return response_file
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

