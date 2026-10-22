from flask import Flask, render_template, request
if __package__:
    from .model import model as m
else:
    import model.model as m
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024
features = m.features

@app.route('/', methods=['GET','POST'])
def hello():
    if request.method == 'GET':
        return render_template('index.html', sale=None)
    if any(not request.form.get(key, '').strip() for key in features):
        return render_template('index.html', error='Complete all 24 fields.', sale=None), 400
    try:
        price = m.predict_pipe([request.form[key] for key in features])
    except ValueError as exc:
        return render_template('index.html', error=str(exc), sale=None), 400
    except RuntimeError:
        app.logger.exception('Model unavailable')
        return render_template('index.html', error='The model is unavailable; check the dependency and artifact installation.', sale=None), 503
    return render_template('index.html', sale=price)

if __name__ == '__main__':
    app.run(debug=False)
