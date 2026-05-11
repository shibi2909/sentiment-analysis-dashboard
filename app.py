from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.sql.types import IntegerType
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Spark Session
spark = SparkSession.builder \
    .appName("SentimentAnalysisDashboard") \
    .master("local[*]") \
    .getOrCreate()

@app.route('/', methods=['GET', 'POST'])

def home():

    # Default dataset
    file_path = 'data/smallreviews.csv'

    # Upload dataset
    if request.method == 'POST':

        uploaded_file = request.files['file']

        if uploaded_file:

            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                uploaded_file.filename
            )

            uploaded_file.save(file_path)

    # -------------------------
    # READ DATASET
    # -------------------------

    try:

        # Try comma-separated CSV
        df = spark.read.csv(

            file_path,

            header=True,

            inferSchema=False,

            multiLine=True,

            escape='"'
        )

        # If only one column detected,
        # try tab-separated file
        if len(df.columns) == 1:

            df = spark.read.csv(

                file_path,

                header=True,

                inferSchema=False,

                sep='\t'
            )

    except Exception as e:

        return f"<h2>Error Reading File:<br>{e}</h2>"

    print("DATASET COLUMNS:")
    print(df.columns)

    # -------------------------
    # POSSIBLE COLUMN NAMES
    # -------------------------

    possible_text_columns = [

        'Text',
        'text',

        'Review',
        'review',

        'Content',
        'content',

        'reviews.text',
        'reviews.title',

        'Summary',
        'summary',

        'comment',
        'Comment',

        'feedback',
        'Feedback'
    ]

    possible_score_columns = [

        'Score',
        'score',

        'Rating',
        'rating',

        'Stars',
        'stars',

        'Liked',
        'liked',

        'reviews.rating',

        'ratings',
        'Ratings',

        'rate',
        'Rate'
    ]

    # -------------------------
    # DETECT TEXT COLUMN
    # -------------------------

    text_column = None

    for c in possible_text_columns:

        if c in df.columns:

            text_column = c
            break

    # -------------------------
    # DETECT SCORE COLUMN
    # -------------------------

    score_column = None

    for c in possible_score_columns:

        if c in df.columns:

            score_column = c
            break

    # -------------------------
    # ERROR HANDLING
    # -------------------------

    if text_column is None or score_column is None:

        return f"""

        <h2 style='text-align:center;
                   margin-top:50px;'>

        Dataset columns not detected.

        <br><br>

        Found Columns:

        <br><br>

        {df.columns}

        </h2>

        """

    # -------------------------
    # CONVERT SCORE TO INTEGER
    # -------------------------

    df = df.withColumn(

        "RatingNumber",

        col(f"`{score_column}`").cast(IntegerType())
    )

    # Remove invalid rows
    df = df.filter(
        col("RatingNumber").isNotNull()
    )

    # -------------------------
    # SENTIMENT CLASSIFICATION
    # -------------------------

    # For restaurant datasets (0/1)
    if score_column.lower() == "liked":

        df = df.withColumn(

            "Sentiment",

            when(
                col("RatingNumber") == 1,
                "Positive"
            ).otherwise("Negative")
        )

    # For normal rating datasets (1-5)
    else:

        df = df.withColumn(

            "Sentiment",

            when(
                col("RatingNumber") >= 4,
                "Positive"
            )

            .when(
                col("RatingNumber") == 3,
                "Neutral"
            )

            .otherwise("Negative")
        )

    # -------------------------
    # STANDARDIZE COLUMN NAMES
    # -------------------------

    df = df.withColumn(

        "Text",

        col(f"`{text_column}`")
    )

    df = df.withColumn(

        "Score",

        col("RatingNumber")
    )

    # Keep only required columns
    df = df.select(

        "Text",

        "Score",

        "Sentiment"
    )

    # Limit rows for dashboard speed
    df = df.limit(100)

    # Convert to Pandas
    pandas_df = df.toPandas()

    # -------------------------
    # SENTIMENT COUNTS
    # -------------------------

    sentiment_counts = pandas_df[
        'Sentiment'
    ].value_counts().to_dict()

    # -------------------------
    # RENDER DASHBOARD
    # -------------------------

    return render_template(

        'index.html',

        reviews=pandas_df.to_dict(
            orient='records'
        ),

        sentiment_counts=sentiment_counts
    )

# -------------------------
# RUN APP
# -------------------------

if __name__ == '__main__':

    app.run(debug=True)