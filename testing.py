import requests
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Function to send a POST request to the API
def send_api_request(api_url, test_case):
    start_time = time.time()  # Record start time
    response = requests.post(api_url, json=test_case)
    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency
    return response.status_code, latency

# Main function for testing and performance tracking
def main():
    # API URL (replace with your AWS Elastic Beanstalk URL)
    api_url = "http://ece444pra-env.eba-rqpemdmz.us-east-2.elasticbeanstalk.com/predict"
    
    # Test cases (two fake news, two real news)
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

# Function to generate boxplot from CSV
def generate_boxplot():
    # Load the CSV file
    df = pd.read_csv("latency_results.csv")

    # Create a boxplot for latency by test case
    plt.figure(figsize=(10, 6))
    df.boxplot(column="Latency (seconds)", by="Test Case", grid=False)

    # Title and labels
    plt.title("API Latency Boxplot by Test Case")
    plt.suptitle("")  # Suppress default title
    plt.xlabel("Test Case")
    plt.ylabel("Latency (seconds)")

    # Save the plot to a file
    plt.savefig("latency_boxplot.png")
    print("Boxplot saved to 'latency_boxplot.png'.")

    # Show the plot
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()
