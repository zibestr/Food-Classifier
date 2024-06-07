from src.web._server import app, clear_uploads


if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=8082)
    except Exception as e:
        clear_uploads()
        print(e)
