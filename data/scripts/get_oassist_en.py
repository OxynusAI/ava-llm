
import json

from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

def build_message_trees(messages):
    """Organize messages into message trees based on their relationships"""
    # Group messages by their tree_id
    trees = defaultdict(list)
    for msg in messages:
        trees[msg["message_tree_id"]].append(msg)
    
    return trees

def build_conversation_from_tree(tree_messages):
    """Build all valid conversation pairs from a message tree"""
    # First, organize messages by their ID for easy lookup
    msg_by_id = {msg["message_id"]: msg for msg in tree_messages}
    
    # Identify root messages (those without a parent or with null parent)
    root_messages = [msg for msg in tree_messages if msg.get("parent_id") is None]
    
    if not root_messages:
        return []
    
    # We'll collect all valid conversations here
    valid_conversations = []
    
    # Process each root message
    for root in root_messages:
        if root["role"] == "prompter" and root["lang"] == "en":
            # Start building conversation paths from this root
            build_conversation_paths(root, msg_by_id, [], valid_conversations)
    
    return valid_conversations

def build_conversation_paths(current_msg, msg_by_id, current_path, valid_conversations):
    """Recursively build all valid conversation paths from the current message"""
    # Add current message to the path
    current_path.append(current_msg)
    
    # Find all direct children of this message
    children = [msg for msg in msg_by_id.values() if msg.get("parent_id") == current_msg["message_id"]]
    
    if not children:
        # This is a leaf node - check if we have a valid conversation
        process_conversation_path(current_path, valid_conversations)
        return
    
    # Process English assistant responses first (we want English conversations)
    en_assistants = [c for c in children if c["role"] == "assistant" and c["lang"] == "en"]
    
    if len(current_path) % 2 == 1 and en_assistants:  # We expect an assistant after a prompter
        # Continue the conversation with each possible assistant response
        for assistant_msg in en_assistants:
            build_conversation_paths(assistant_msg, msg_by_id, current_path.copy(), valid_conversations)
    else:
        # Either we have no English assistant responses or we're not at the right point
        # Just process what we have so far
        process_conversation_path(current_path, valid_conversations)

def process_conversation_path(path, valid_conversations):
    """Process a conversation path to extract valid user-assistant pairs"""
    # Check if we have a valid alternating conversation
    if len(path) < 2:
        return
    
    # Ensure it's a valid alternating prompter-assistant conversation
    for i in range(len(path)):
        expected_role = "prompter" if i % 2 == 0 else "assistant"
        if path[i]["role"] != expected_role or path[i]["lang"] != "en":
            # Invalid path or non-English content
            return
    
    # Convert to the desired format
    formatted_conversation = []
    for i in range(0, len(path), 2):
        if i + 1 < len(path):  # Ensure we have both user and assistant
            pair = [
                {"role": "user", "content": path[i]["text"]},
                {"role": "assistant", "content": path[i+1]["text"]}
            ]
            formatted_conversation.append(pair)
    
    if formatted_conversation:
        valid_conversations.append(formatted_conversation)

def process_oasst_dataset():
    """Main function to download and process the OpenAssistant dataset"""
    print("Loading OpenAssistant dataset...")
    dataset = load_dataset("OpenAssistant/oasst1")
    
    print("Processing messages...")
    # Convert dataset to a list of dictionaries for easier processing
    train_messages = [dict(message) for message in dataset["train"]]
    
    # Filter to only include English messages to reduce processing time
    en_messages = [msg for msg in train_messages if msg["lang"] == "en"]
    print(f"Found {len(en_messages)} English messages out of {len(train_messages)} total messages")
    
    # Group messages by their tree_id
    message_trees = build_message_trees(en_messages)
    print(f"Found {len(message_trees)} message trees with English content")
    
    # Process each tree to extract conversations
    all_conversations = []
    for tree_id, messages in tqdm(message_trees.items(), desc="Processing trees"):
        conversations = build_conversation_from_tree(messages)
        all_conversations.extend(conversations)
    
    # Save to file
    output_file = "oasst1_en_conv.json"
    print(f"Saving {len(all_conversations)} conversations to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Saved {len(all_conversations)} English conversations.")
    # Print a sample if available
    if all_conversations:
        print(f"Sample conversation:\n{json.dumps(all_conversations[0], indent=2)}")
    else:
        print("No conversations found")

if __name__ == "__main__":
    process_oasst_dataset()