import os
import pandas as pd
# import transformers to use the standard summarizer
from transformers import pipeline, AutoTokenizer


def summarize_transcript(transcript):
    # Load the lightweight summarization model
    # require Tensor FLow weights since the model is not available in PyTorch
    print("Loading summarization model...")
    summarizer = pipeline("summarization", model="t5-small", device=-1)  # Use device=0 for GPU, or -1 for CPU
    # Using tokenizer to build chuncks of 256 size for better T5 performance
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Split the transcript into smaller chunks
    max_chunk_size = 256
    tokens = tokenizer.encode(transcript, return_tensors="pt", truncation=False)[0]
    chunks = [tokens[i:i + max_chunk_size] for i in range(0, len(tokens), max_chunk_size)]

    # Summarize each chunk
    print("Generating summary...")
    summaries = []
    len_chunks = len(chunks)
    for chunk in chunks:
        print(f"Processing chunk {len(summaries) + 1} of {len_chunks}")
        # Decode the chunk back to text
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarizer(chunk_text, max_length=200, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Combine all chunk summaries into a single summary
    return " ".join(summaries)



def pull_transcripts(path):
    df_summaries = pd.DataFrame(columns=['company_name', 'summary'])
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                transcript = file.read()
                company_name = filename.split('.')[0]  # Assuming the filename is the company name
                print("Send Transcript")
                summary = summarize_transcript(transcript)
    
                # Append the company name and summary to the DataFrame
                df_summaries = df_summaries.append({'company_name': company_name, 'summary': summary}, ignore_index=True)

    df_summaries.to_csv('transcript_summaries.csv', index=False)

if __name__ == "__main__":
    # Specify the path to the directory containing the transcript files
    transcripts_folder = "Transcripts"  # Path to your transcripts folder
    pull_transcripts(transcripts_folder)