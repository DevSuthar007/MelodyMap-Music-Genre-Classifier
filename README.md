# 🎵 MelodyMap: Music Genre Classifier

MelodyMap predicts the genre of a song from its audio features. Upload a CSV
with the 57 features of one song, or upload an audio file, and the app shows
the predicted genre, the confidence and the probability of all 10 genres.

**Live demo:** https://huggingface.co/spaces/samya1234/music-genre-classifier

## Features
- Dual input: 57-feature CSV (one song) or audio file (features extracted with librosa)
- Top genre, confidence score and probability bars for all genres
- Waveform, mel-spectrogram and chroma plots for uploaded audio
- Genre Explorer: history, instruments, rhythm and listening tips per genre

## Dataset
GTZAN: 1,000 clips (30 s each), 10 genres, 100 clips per genre.
Genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.
57 features: chroma, RMS, spectral centroid/bandwidth/roll-off, zero-crossing
rate, harmonic and percussive components, tempo and MFCC 1-20.

## Method
Dataset → cleaning → features/labels separation → stratified 80/20 split →
scaling (fit on train only) → KNN / Decision Tree / Random Forest / SVM →
evaluation → best model saved as a scikit-learn Pipeline (SimpleImputer + classifier).

## Results
| Model         | Accuracy | Precision | Recall | F1    |
|---------------|----------|-----------|--------|-------|
| SVM           | 0.765    | 0.769     | 0.765  | 0.764 |
| Random Forest | 0.690    | 0.691     | 0.690  | 0.687 |
| KNN           | 0.685    | 0.703     | 0.685  | 0.685 |
| Decision Tree | 0.525    | 0.555     | 0.525  | 0.528 |

Best model: SVM (76.5% accuracy).

## Run locally
    pip install -r requirements.txt
    python app.py

## Limitations
- GTZAN is small and old, with some repeated and mislabeled clips
- Only 10 genres (no Indian or other modern genres)
- One genre per song, though real songs often mix genres

## Future work
CNN on mel-spectrograms, a larger dataset (e.g., FMA), real-time prediction,
Indian genres and ragas, mood detection, similarity search, multi-label genres.

## Team
Samya Tiwari, Rachita Prasad, Dev Suthar, Aeshva Rao, Raj Vani
Unitedworld Institute of Technology, Karnavati University, Gandhinagar, India

## References
Tzanetakis & Cook, "Musical genre classification of audio signals,"
IEEE Trans. Speech Audio Process., 2002.
