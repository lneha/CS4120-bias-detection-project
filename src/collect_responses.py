"""
LLM Response Collection System
Collects responses from Claude, GPT-4, and Gemini (FREE) for bias analysis
CS4120 Natural Language Processing - Northeastern University
"""

import json
import os
import time
from datetime import datetime
import anthropic
import openai
from google import generativeai as genai

class LLMResponseCollector:
    def __init__(self):
        """Initialize all three LLM API clients"""
        self.claude_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        
        self.openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        print("Initialized Claude Sonnet 4")
        print("Initialized GPT-4")
        print("Initialized Gemini 1.5 Flash (FREE)")
    
    def query_claude(self, prompt):
        """Query Claude Sonnet 4"""
        try:
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def query_gpt4(self, prompt):
        """Query GPT-4"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def query_gemini(self, prompt):
        """Query Gemini 1.5 Flash (FREE)"""
        try:
            response = self.gemini_model.generate_content(prompt)
            
            if hasattr(response, 'prompt_feedback'):
                if response.prompt_feedback.block_reason:
                    return f"BLOCKED: {response.prompt_feedback.block_reason}"
            
            return response.text
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def process_topic_file(self, input_path, output_path):
        """Process a single topic JSON file and collect all responses"""
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(input_path)}")
        print(f"{'='*70}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_prompts = sum(len(items) for items in data.values())
        current = 0
        
        for category_name, prompt_list in data.items():
            print(f"\n  Category: {category_name} ({len(prompt_list)} prompts)")
            
            for item in prompt_list:
                current += 1
                prompt_text = item['prompt']
                
                print(f"\n    [{current}/{total_prompts}] Prompt: {prompt_text[:60]}...")
                
                print("      Querying Claude...", end=" ")
                item['responses']['claude'] = self.query_claude(prompt_text)
                print("Done")
                time.sleep(1)
                
                print("      Querying GPT-4...", end=" ")
                item['responses']['gpt4'] = self.query_gpt4(prompt_text)
                print("Done")
                time.sleep(1)
                
                print("      Querying Gemini...", end=" ")
                item['responses']['gemini'] = self.query_gemini(prompt_text)
                print("Done")
                time.sleep(2)
        
        output_data = {
            'metadata': {
                'source_file': os.path.basename(input_path),
                'collection_date': datetime.now().isoformat(),
                'total_prompts': total_prompts,
                'models': ['claude-sonnet-4', 'gpt-4', 'gemini-1.5-flash']
            },
            'data': data
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved: {output_path}")
        return output_path
    
    def collect_all_topics(self):
        """Collect responses for all five topics"""
        topics = ['climate_policy', 'gun_control', 'immigration', 'mental_health', 'tariffs']
        completed = []
        
        for topic in topics:
            input_path = f'../data/prompts/{topic}.json'
            output_path = f'../data/responses/{topic}_responses.json'
            
            if not os.path.exists(input_path):
                print(f"Warning: {input_path} not found, skipping...")
                continue
            
            try:
                completed_file = self.process_topic_file(input_path, output_path)
                completed.append(completed_file)
            except Exception as e:
                print(f"Error processing {topic}: {e}")
                continue
        
        return completed

def verify_api_keys():
    """Verify all required API keys are present"""
    required = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']
    missing = [k for k in required if not os.environ.get(k)]
    
    if missing:
        print("\nMissing API keys:")
        for k in missing:
            print(f"  - {k}")
        print("\nSet with:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'")
        return False
    
    return True

def main():
    print("\n" + "="*70)
    print("LLM Response Collection System")
    print("CS4120 NLP - Bias Detection Project")
    print("="*70)
    
    if not verify_api_keys():
        return
    
    os.makedirs('../data/prompts', exist_ok=True)
    os.makedirs('../data/responses', exist_ok=True)
    
    collector = LLMResponseCollector()
    completed = collector.collect_all_topics()
    
    print("\n" + "="*70)
    print(f"Collection Complete: {len(completed)}/5 files processed")
    print("="*70)
    
    if len(completed) < 5:
        print("\nSome files were not processed. Check errors above.")

if __name__ == "__main__":
    main()
