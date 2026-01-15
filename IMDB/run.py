from loader import fetch_and_extract, preproc, build_model, save_model
from upload import upload_for_eval

def main():
    fetch_and_extract('https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip')
    data_frame = preproc('train.json')
    model = build_model(data_frame)
    save_model(model, 'imdb.joblib')
    upload_for_eval('http://10.129.237.126:5000/api/upload', 'imdb.joblib')

if __name__ == '__main__':
    main()
