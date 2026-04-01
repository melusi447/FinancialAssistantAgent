"""
Core Prompt Service
Robust prompt template management with caching and error handling
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

from config import config

logger = logging.getLogger(__name__)

class PromptService:
    """Robust prompt service with caching and error handling"""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        self.prompts_dir = prompts_dir or config.PROMPTS_DIR
        self._ensure_prompts_exist()
        self._cache = {}
    
    def _ensure_prompts_exist(self):
        """Ensure default prompt files exist"""
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Create default system prompt if it doesn't exist
        system_prompt_path = os.path.join(self.prompts_dir, "system_prompt.txt")
        if not os.path.exists(system_prompt_path):
            self._create_default_system_prompt(system_prompt_path)
        
        # Create default retrieval prompt if it doesn't exist
        retrieval_prompt_path = os.path.join(self.prompts_dir, "retrieval_prompt.txt")
        if not os.path.exists(retrieval_prompt_path):
            self._create_default_retrieval_prompt(retrieval_prompt_path)
    
    def _create_default_system_prompt(self, file_path: str):
        """Create default system prompt"""
        default_prompt = """You are FinanceBot, a professional AI financial assistant.

Your expertise includes:
- Investment analysis and portfolio management
- Risk assessment and financial planning
- Market analysis and economic trends
- Personal finance and budgeting advice
- Retirement planning and wealth management

Guidelines:
- Provide accurate, structured financial advice based on current best practices
- Always include relevant risk considerations
- Avoid speculation and admit when information is uncertain
- Never provide legal, tax, or medical advice
- Be clear about the limitations of your knowledge
- Use data-driven insights when possible
- Maintain a professional, helpful tone

Response format:
- Start with a clear, direct answer
- Provide supporting reasoning and context
- Include relevant risk factors
- Suggest next steps or additional considerations when appropriate"""
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(default_prompt)
            logger.info(f"Created default system prompt: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create system prompt: {e}")
    
    def _create_default_retrieval_prompt(self, file_path: str):
        """Create default retrieval prompt"""
        default_prompt = """You are FinanceBot, a professional AI financial assistant.

Your expertise includes:
- Investment analysis and portfolio management
- Risk assessment and financial planning
- Market analysis and economic trends
- Personal finance and budgeting advice
- Retirement planning and wealth management

Use the following retrieved financial context to inform your response:

{context}

User Question: {user_input}

Based on the context above and your financial expertise, provide a comprehensive answer that:
- Directly addresses the user's question
- Incorporates relevant information from the context
- Includes appropriate risk considerations
- Provides actionable insights when possible
- Maintains professional financial advisory standards

If the context doesn't contain relevant information, rely on your general financial knowledge while noting any limitations."""
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(default_prompt)
            logger.info(f"Created default retrieval prompt: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create retrieval prompt: {e}")
    
    @lru_cache(maxsize=32)
    def _load_template(self, filename: str) -> str:
        """Load and cache prompt template"""
        try:
            file_path = os.path.join(self.prompts_dir, filename)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                template = f.read()
            
            return template
            
        except Exception as e:
            logger.error(f"Error loading prompt template {filename}: {e}")
            return self._get_fallback_prompt(filename)
    
    def _get_fallback_prompt(self, filename: str) -> str:
        """Get fallback prompt when template loading fails"""
        if "retrieval" in filename.lower():
            return """You are a financial assistant. Use the following context to answer the user's question:

Context: {context}

User Question: {user_input}

Please provide a helpful financial response based on the context and your knowledge."""
        else:
            return "You are a financial assistant. Provide helpful and accurate financial advice."
    
    def load_prompt(self, filename: str, **kwargs) -> str:
        """Load a prompt template and inject variables"""
        try:
            # Load template (cached)
            template = self._load_template(filename)
            
            # Inject variables into template
            try:
                formatted_prompt = template.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing variable {e} in prompt template {filename}")
                # Fill missing variables with empty strings
                formatted_prompt = template.format(**{k: v for k, v in kwargs.items() if k in template})
            
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Error loading prompt {filename}: {e}")
            return self._get_fallback_prompt(filename)
    
    def get_available_prompts(self) -> list:
        """Get list of available prompt files"""
        try:
            prompt_files = []
            for file_path in Path(self.prompts_dir).glob("*.txt"):
                prompt_files.append(file_path.name)
            return sorted(prompt_files)
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return []
    
    def create_custom_prompt(self, filename: str, content: str) -> bool:
        """Create a new custom prompt file"""
        try:
            file_path = os.path.join(self.prompts_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Clear cache for this file
            self._load_template.cache_clear()
            
            logger.info(f"Created custom prompt: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error creating custom prompt {filename}: {e}")
            return False
    
    def update_prompt(self, filename: str, content: str) -> bool:
        """Update an existing prompt file"""
        try:
            file_path = os.path.join(self.prompts_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Clear cache for this file
            self._load_template.cache_clear()
            
            logger.info(f"Updated prompt: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error updating prompt {filename}: {e}")
            return False
    
    def delete_prompt(self, filename: str) -> bool:
        """Delete a prompt file"""
        try:
            file_path = os.path.join(self.prompts_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                # Clear cache
                self._load_template.cache_clear()
                logger.info(f"Deleted prompt: {filename}")
                return True
            else:
                logger.warning(f"Prompt file not found: {filename}")
                return False
        except Exception as e:
            logger.error(f"Error deleting prompt {filename}: {e}")
            return False
    
    def get_prompt_info(self) -> Dict[str, Any]:
        """Get information about prompt service"""
        return {
            "prompts_dir": self.prompts_dir,
            "available_prompts": self.get_available_prompts(),
            "cache_size": len(self._cache)
        }

# Global prompt service instance
prompt_service = PromptService()

# Convenience function for backward compatibility
def load_prompt(filename: str, **kwargs) -> str:
    """Convenience function to load a prompt template"""
    return prompt_service.load_prompt(filename, **kwargs)



