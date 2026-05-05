To run this project u must have to download the trained model from this link  https://drive.google.com/drive/folders/1iQ51FhYBBGk2koe7G4Vcs15E3KOzoLca?usp=drive_link

Make sure to create the folder "checkpoints_video_v2" and keep the downloaded model in this folder

clone the repo using 
    git clone https://github.com/mahender9390/Deepfake_Faceswap_Detector.git


Then go to the Folder in terminal and create a virtual environment
      python -m venv venv

Now Activate the virtual enivironment using 
      venv\Scripts\activate

After that run 
    pip install -r requirements.txt



TO RUN FRONTEND
  1.open cmd or vs code 
  2.Go to virtual environment
  3.run cd .\frontend\ 
  4.npm install
  5.npm run dev

TO RUN BACKEND
  1.open cmd or vs code
  2.go to virtual environment
  3.run uvicorn backend.main:app --host 0.0.0.0 --port 8000

To view ur project go to http://localhost:3000
