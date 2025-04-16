# This script uses the same sumarization as "Summarize.PY" but uses an intermediate step 
# to read transcripts and break it down into key parts, then summarizes each part.
# Using the T5-SMALL MODEL

import os
import pandas as pd
import time
# import transformers to use the standard summarizer
from transformers import pipeline, AutoTokenizer


def summarize_transcript(transcript):
    # Load the lightweight summarization model
    print("Loading summarization model...")
    summarizer = pipeline("summarization", model="t5-small", device=-1)  # Use device=0 for GPU, or -1 for CPU
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Split the transcript into smaller chunks
    max_chunk_size = 500  # Adjusted to fit within the T5 model's context window
    tokens = tokenizer.encode(transcript, return_tensors="pt", truncation=False)[0]
    chunks = [tokens[i:i + max_chunk_size] for i in range(0, len(tokens), max_chunk_size)]

    # Summarize each chunk
    print("Generating intermediate summaries...")
    intermediate_summary = []
    len_chunks = len(chunks)
    for chunk in chunks:
        print(f"Processing chunk {len(intermediate_summary) + 1} of {len_chunks}")
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarizer(chunk_text, max_length=70, min_length=30, do_sample=False)
        intermediate_summary.append(summary[0]['summary_text'])

    # Combine intermediate summaries into a single string
    combined_summary = " ".join(intermediate_summary)

    # Split the combined summary into smaller chunks for final summarization
    print("Generating final summary...")
    combined_tokens = tokenizer.encode(combined_summary, return_tensors="pt", truncation=False)[0]
    combined_chunks = [combined_tokens[i:i + 512] for i in range(0, len(combined_tokens), 512)]

    final_summary_parts = []
    for chunk in combined_chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarizer(chunk_text, max_length=70, min_length=30, do_sample=False)
        final_summary_parts.append(summary[0]['summary_text'])

    # Combine all final summary parts into a single summary
    final_summary = " ".join(final_summary_parts)
    print(final_summary)
    return final_summary

def pull_transcripts(path):
    df_summaries = pd.DataFrame(columns=['company_name', 'summary'])
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                transcript = file.read()
                company_name = filename.split('.')[0]  # Assuming the filename is the company name
                
                start_time = time.time()
                print("Send Transcript")
                summary = summarize_transcript(transcript)
                # Calculate the time taken to summarize
                end_time = time.time()
                time_taken = end_time - start_time
                hours, remainder = divmod(time_taken, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_taken_str = f"{int(hours)}H:{int(minutes)}M:{int(seconds)}S"

                # Append the company name and summary to the DataFrame using pd.concat
                new_row = pd.DataFrame({'company_name': [company_name], 
                                        'summary': [summary],
                                        'time': [time_taken_str]})
                df_summaries = pd.concat([df_summaries, new_row], ignore_index=True)

    # Create Results Folder if it doesnt' exist
    if not os.path.exists('Results'):
        os.makedirs('Results')
    # Save the summaries to the Results directory
    df_summaries.to_csv(os.path.join("Results", '2step_transcript_summaries.csv'), index=False)

if __name__ == "__main__":
    # Specify the path to the directory containing the transcript files
    transcripts_folder = "Transcripts"  # Path to your transcripts folder
    pull_transcripts(transcripts_folder)