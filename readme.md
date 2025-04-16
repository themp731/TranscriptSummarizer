# Transcript Summarizer

The `summarize.py` script is a Python program designed to generate concise summaries of textual transcripts using a lightweight Hugging Face NLP model. It processes transcript files stored in a directory, summarizes their content, and outputs the summaries in a CSV file for easy access and analysis.

The `summarize_twostep.py` script introduces a hierarchical summarization approach to handle large transcripts more effectively. This method processes transcripts in two steps to generate a more concise and cohesive summary.

**When to Use:**  
Use `summarize.py` for shorter transcripts or when you need a quick summary.
Use `summarize_twostep.py` for longer transcripts (e.g., 30,000+ words) to ensure better summarization quality.



## Features

- **Transcript Summarization**: Breaks down large transcript files into manageable chunks, summarizes each chunk, and combines the results into a cohesive summary.
- **Batch Processing**: Processes multiple transcript files from a specified directory.
- **Hugging Face Integration**: Utilizes the `transformers` library for state-of-the-art text summarization via the T5-small model.
- **CSV Output**: Saves the generated summaries along with the corresponding transcript file names in a CSV file.

---

## How It Works

1. **Transcript Splitting**:
   - The script uses the Hugging Face `AutoTokenizer` to tokenize and split the input transcript into smaller chunks of 256 tokens for optimal performance with the T5-small summarization model.

2. **Summarization**:
   - The Hugging Face `pipeline` is used to load the T5-small summarization model.
   - Each chunk of the transcript is summarized using the `summarization` pipeline.
   - Summaries for all chunks are combined into a single cohesive summary.

3. **Batch Processing**:
   - The `pull_transcripts` function iterates through a directory of `.txt` files.
   - Each file is read, summarized, and its results are stored in a Pandas DataFrame.

4. **Output**:
   - The summaries are saved to a file named `transcript_summaries.csv` for further analysis.

---

## Requirements

To run the script, ensure you have the following installed:

- Python 3.7 or later
- Required Python libraries:
  - `transformers` (Hugging Face library)
  - `torch` (needed for the tokenizer and model)
  - `pandas`

You can install the required libraries using pip:
`bash`
`pip install transformers torch pandas`


## Setup Instructions
Clone the Repository
Clone this repository to your local system using the following command:

bash
git clone https://github.com/themp731/TranscriptSummarizer.git
cd TranscriptSummarizer
Prepare the Transcripts

Create a folder named Transcripts in the root directory of the repository.
Add .txt files containing the transcripts you want to summarize into this folder.
Each file should be named after the company or entity to which the transcript belongs (e.g., companyA.txt).
Install Dependencies
Install the required Python libraries using pip:

bash
pip install transformers torch pandas
Run the Script
Execute the script to summarize the transcripts:

bash
python summarize.py
View the Results
The summarized data will be saved in a file named transcript_summaries.csv in the root directory of the repository.

## Notes
**Running on GPU:**
By default, the script is configured to run on a CPU (device=-1). If you have a GPU and the necessary drivers installed, modify the script to use device=0:

**Python**
```python
summarizer = pipeline("summarization", model="t5-small", device=0)
```
**Transcript Format:**
Ensure that the transcript files are plain text files (.txt) and are stored in the Transcripts directory.

**Performance Considerations:**
For large datasets or long transcripts, using a GPU is highly recommended to speed up the summarization process.
summarizer = pipeline("summarization", model="t5-small", device=0)

## Troubleshooting
**Model Download Issues:**
If the script fails to load the T5-small model, ensure you have an active internet connection. The model weights are downloaded from Hugging Face's repository during the first execution.

**Virtual Environment Setup:**
For a clean and isolated environment, consider setting up a virtual environment:

bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install transformers torch pandas
Error Handling:
If you encounter issues with large files or memory errors, try splitting the input transcript files into smaller pieces manually before running the script.

## Output Example
The generated transcript_summaries.csv file will look like this:

company_name	summary Time
companyA	"This is a concise summary of the transcript for companyA..."   0H5M21S
companyB	"This is a concise summary of the transcript for companyB..."   0H4M15S

