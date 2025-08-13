# code to stream the price of the ticker. You can bring this function into the app and replace
# with the "selected_stock" var for streaming from local or can host virtually by replacing the local server port

import time
import json
from kafka import KafkaProducer
import yfinance as yf


def get_price(symbol):
    data = yf.Ticker(symbol).history(period="1d", interval="1m")  # get data from the last day every minute
    if not data.empty:
        price = round(data['Close'].iloc[-1], 2)  # -1 = last row
        return price
    return None


def main():
    symbol = "AAPL"  # dummy symbol - replace with selected stock in app
    kafka_broker = "localhost:9092"  # address of kafka server
    topic = "stock-prices"  # topic = message channel / category for Kafka

    producer = KafkaProducer(
        bootstrap_servers=kafka_broker,  # tells producer where to connect
        value_serializer=lambda v: json.dumps(v).encode('utf-8')  # converts the python object to a json string and turns into bytes (utf-8). Kafka uses bytes.
    )

    print(f"Streaming {symbol} stock price to Kafka topic '{topic}' every 10 seconds...")

    while True:
        price = get_price(symbol)
        if price:
            message = {
                "symbol": symbol,
                "price": price,
                "timestamp": time.time()
            }
            producer.send(topic, message)
            print(f"Sent: {message}")
            producer.flush()
        else:
            print("Failed to get price")

        time.sleep(5)


if __name__ == "__main__":
    main()
