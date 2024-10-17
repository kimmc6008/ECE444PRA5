import requests
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt

def send_api_request(api_url, test_case):
    start_time = time.time()
    response = requests.post(api_url, json=test_case)
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency

def main():
    api_url = "http://ece444pra-env.eba-rqpemdmz.us-east-2.elasticbeanstalk.com/predict"
    
    test_data = [
        {"text": "Cats can fly now"},  # Fake news
        {"text": "1 plus 1 is 5028402934"},  # Fake news
        {"text": "UofT has a program called Engineering Science"},  # Real news
        {"text": "Toronto is in Canada"}   # Real news
    ]
    
    results = []

    # Perform 100 API calls for each test case
    for i in range(100):  
        for case_index, test_case in enumerate(test_data):
            status_code, latency = send_api_request(api_url, test_case)
            results.append([case_index, status_code, latency])
            print(f"Test Case {case_index}, Response: {status_code}, Latency: {latency:.4f} seconds")
    
    # Write results to a CSV file
    with open("latency_results.csv", "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Test Case", "Response Code", "Latency (seconds)"])  # Header
        csvwriter.writerows(results)
    
    print("Latency results saved to 'latency_results.csv'.")

    # Generate boxplot
    generate_boxplot()

def generate_boxplot():
    df = pd.read_csv("latency_results.csv")

    plt.figure(figsize=(10, 6))
    df.boxplot(column="Latency (seconds)", by="Test Case", grid=False)

    plt.title("API Latency Boxplot by Test Case")
    plt.suptitle("")
    plt.xlabel("Test Case")
    plt.ylabel("Latency (seconds)")

    plt.savefig("latency_boxplot.png")
    print("Boxplot saved to 'latency_boxplot.png'.")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
