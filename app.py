from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def getFruits():
    return "Fruits!"

if __name__ == '__main__':
    app.run()