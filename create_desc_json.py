import argparse
from pathlib import Path
import json
import librosa
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_audio_duration(audio_path: Path) -> float:
    """
    Calculate the duration of an audio file in seconds.

    Args:
        audio_path (Path): Path to the audio file.

    Returns:
        float: Duration in seconds.
    """
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.error(f"Error calculating duration for {audio_path}: {e}")
        return 0.0

def read_transcriptions(transcript_path: Path) -> Dict[str, str]:
    """
    Read transcriptions from a .txt file.

    Args:
        transcript_path (Path): Path to the transcription file.

    Returns:
        Dict[str, str]: Mapping of file IDs to transcriptions.
    """
    transcriptions = {}
    try:
        with transcript_path.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcriptions[file_id] = text.lower()
    except Exception as e:
        logger.error(f"Error reading {transcript_path}: {e}")
    return transcriptions

def process_directory(data_dir: Path) -> List[Dict]:
    """
    Process the data directory to collect audio metadata.

    Args:
        data_dir (Path): Root directory containing group/speaker subdirectories.

    Returns:
        List[Dict]: List of metadata entries with audio path, duration, and transcription.
    """
    metadata = []
    
    logger.info(f"Scanning directory: {data_dir}")
    
    # Iterate over groups
    for group_dir in data_dir.iterdir():
        if group_dir.is_dir() and not group_dir.name.startswith('.'):
            # Iterate over speakers
            for speaker_dir in group_dir.iterdir():
                if speaker_dir.is_dir() and not speaker_dir.name.startswith('.'):
                    logger.info(f"Processing {group_dir.name}/{speaker_dir.name}")
                    
                    # Look for transcription file
                    transcript_file = speaker_dir / f"{group_dir.name}-{speaker_dir.name}.trans.txt"
                    if not transcript_file.exists():
                        logger.warning(f"Transcription file not found: {transcript_file}")
                        continue
                    
                    # Read transcriptions
                    transcriptions = read_transcriptions(transcript_file)
                    
                    # Process audio files
                    for audio_file in speaker_dir.glob("*.wav"):
                        file_id = audio_file.stem
                        if file_id in transcriptions:
                            duration = calculate_audio_duration(audio_file)
                            metadata.append({
                                "audio_path": str(audio_file),
                                "duration": duration,
                                "transcription": transcriptions[file_id]
                            })
    
    logger.info(f"Collected {len(metadata)} audio files")
    return metadata

def write_jsonl(output_path: Path, metadata: List[Dict]):
    """
    Write metadata to a JSON-line file.

    Args:
        output_path (Path): Path to the output file.
        metadata (List[Dict]): List of metadata entries.
    """
    logger.info(f"Writing output to {output_path}")
    with output_path.open('w', encoding='utf-8') as f:
        for entry in metadata:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate JSON-line metadata for deep-speech training")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the root data directory (organized like LibriSpeech)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSON-line file"
    )
    args = parser.parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    # Process directory and write output
    metadata = process_directory(data_dir)
    write_jsonl(output_path, metadata)

if __name__ == "__main__":
    main()