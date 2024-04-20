import os
import requests
import json
import apikey
import serpapi
import yfinance as yf
from yahooquery import Ticker
import openai
from openai import OpenAI

os.environ["SERPAPI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# This function provides news about the company from google using  "SERP API"
def get_company_news(company_name):
    # Parameters to be used in the API request
    params = {
        "engine": "google", # Search engine used for news retrieval
        "tbm": "nws",   # Indicates news search on Google
        "q": company_name,  # Query parameter: company name for which news is being searched
        "api_key": os.environ["SERPAPI_API_KEY"],   # API key obtained from environment variables
    }
    # Sending an HTTP GET request to the SERP API with defined parameters
    response = requests.get('https://serpapi.com/search', params=params)
     # Parsing the response content as JSON
    data = response.json()
    # Extracting and returning the news results from the API response
    return data.get('news_results')


#function is designed to write news details to a file.
def write_news_to_file(news, filename):
    # Open the file in 'write' mode and associate it with the 'file' variable
    with open(filename, 'w') as file:
         # Iterate through each news item in the 'news' list
        for news_item in news:
            # Check if the news item exists (not None)
            if news_item is not None:
                # Extract information from the news item or assign default values if not available
                title = news_item.get('title', 'No title')  # Get news title or set default value
                link = news_item.get('link', 'No link')
                date = news_item.get('date', 'No date')
                 # Write the news item details to the opened file
                file.write(f"Title: {title}\n") # Write title to file
                file.write(f"Link: {link}\n")   # Write link to file
                file.write(f"Date: {date}\n\n")  # Write date to file with extra newline




#function fetches historical market data for a specified company and period
# And saves it to a file named "investment.txt
def get_stock_evolution(company_name, period="1y"):
# Get the stock information for the specified company
 stock = yf.Ticker(company_name)

# Get historical market data for the specified period
 hist = stock.history(period=period)

 # Convert the DataFrame to a string with a specific format
 data_string = hist.to_string()

# Append the string to the "investment.txt" file
 with open("investment.txt", "a") as file:
  file.write(f"\nStock Evolution for {company_name}:\n")
  file.write(data_string)
  file.write("\n")

  # Return the historical market data (DataFrame)
 return hist



#function retrieves various financial statements and valuation measures for a specified company
# using its ticker symbol
def get_financial_statements(ticker):
	# Create a Ticker object using the provided ticker symbol
	company = Ticker(ticker)

	# Get financial data:  Balance Sheet, Cash Flow, Income Statement, and Valuation Measures
	balance_sheet = company.balance_sheet().to_string()
	cash_flow = company.cash_flow(trailing=False).to_string()
	income_statement = company.income_statement().to_string()
	valuation_measures = str(company.valuation_measures)  # This one might already be a dictionary or string

	# Write data to file
	with open("investment.txt", "a") as file:
		file.write("\nBalance Sheet\n")
		file.write(balance_sheet)
		file.write("\nCash Flow\n")
		file.write(cash_flow)
		file.write("\nIncome Statement\n")
		file.write(income_statement)
		file.write("\nValuation Measures\n")
		file.write(valuation_measures)


#This function orchestrates the retrieval and storage of different types of data
# related to a company specified
def get_data(company_name, company_ticker, period="1y", filename="investment.txt"):
    # Get news related to the company
    news = get_company_news(company_name)
    if news:
        # Write news to the specified file
        write_news_to_file(news, filename)
    else:
        print("No news found.")

    # Calling above two functions
    hist = get_stock_evolution(company_ticker)
    try:
        get_financial_statements(company_ticker)
    except:
        hist = "Company not found/ Delisted"
        return hist
    # Return the historical stock data
    return hist


#used to store an instance of the OpenAI client
client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
)


def financial_analyst(request):
    print(f"Received request: {request}")

    # Make an API call to OpenAI GPT-3.5 model for chat completions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[{
            "role":"user",
            "content": f"Given the user request, what is the company name and the company stock ticker ?: {request}?"
        }],
        functions=[{
            "name": "get_data",
            "description": "Get financial data on a specific company for investment purposes",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    },
                    "company_ticker": {
                        "type": "string",
                        "description": "the ticker of the stock of the company"
                    },
                    "period": {
                        "type": "string",
                        "description": "The period of analysis"
                    },
                    "filename": {
                        "type": "string",
                        "description": "the filename to store data"
                    }
                },
                "required": ["company_name", "company_ticker"],
            },
        }],
        function_call={"name": "get_data"},
    )
     # Extract message data from the API response
    message_data = response.choices[0].message


    # Check if the message data has function call information
    if hasattr(message_data, "function_call"):
        # Parse the arguments from a JSON string to a Python dictionary
        arguments = json.loads(message_data.function_call.arguments)
        print(arguments,'\n===========================================================\n')
        company_name = arguments["company_name"]
        company_ticker = arguments["company_ticker"]

         # Retrieve financial data based on the company name and ticker
        hist = get_data(company_name, company_ticker)
        
        # Read content from "investment.txt" file
        with open("investment.txt", "r") as file:
            content = file.read()[:14000]

         # Generate a response using GPT-3.5 model to create an investment thesis
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": request
                },
                message_data,
                {
                    "role": "system",
                    "content": """write a detailed investment thesis to answer
                        the user request as document. Provide numbers to justify
                        your assertions, a lot ideally. Always provide
                        a recommendation to buy the stock of the company
                        or not given the information available. """
                },
                {
                    "role": "assistant",
                    "content": content,
                },
            ],
        )
        # Extract content from the second response
        content_to_save = second_response.choices[0].message.content
        
         # Return the extracted content and financial data
        return (content_to_save,hist)

        



