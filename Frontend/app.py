from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import replicate
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For flash messages

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the file temporarily
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            # Process the image
            process_image(filepath)
            os.remove(filepath)  # Remove the file after processing
            return redirect(url_for('index'))
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg']

@app.route('/run_command', methods=['POST'])
def run_command():
    try:
        result = subprocess.run(["chainlit", "run", "frontend.py"], capture_output=True, text=False)
        flash(f'Command output: {result.stdout}')
    except Exception as e:
        flash(str(e))
    return redirect(url_for('index'))


def process_image(image_path):
    questions = ["Describe the image", "Is the given Image a Urban Landscape or Agricultural ?", 
                 "Is there any water body present in the image?",
                 "Is there any Roads and Transportation Networks visible in the image?"]
    with open(image_path, "rb") as img, open("data.txt", "a") as file:
        print("Processing the image please wait")
        for i, question in enumerate(questions):
            print(question)
            if i == 0:
                output = replicate.run("nvlabs/prismer:e604611dc43bfabc4eb5cda01eab65a491d74910cf5545da2a189718320873b1",
                                       input={"task": "caption", "model_size": "base", "input_image": img, "use_experts": True, "output_expert_labels": True})
                file.write(f"Image Context is :\nAnswer: {output['answer']}\n\n")
            else:
                output = replicate.run("nvlabs/prismer:e604611dc43bfabc4eb5cda01eab65a491d74910cf5545da2a189718320873b1",
                                       input={"task": "vqa", "question": question, "model_size": "base", "input_image": img, "use_experts": True, "output_expert_labels": False})
                print(output)
                file.write(f"Question: {question}\nAnswer: {output['answer']}\n\n")
        flash("Description is saved to 'data.txt'.")

if __name__ == '__main__':
    app.run(debug=True)
