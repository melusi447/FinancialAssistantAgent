"""
Core Database Service
Robust SQLite database management for conversations, feedback, and analytics
"""

import os
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
import threading

from config import config

logger = logging.getLogger(__name__)

class DatabaseService:
    """Robust database service for conversation logging and analytics"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DATABASE_PATH
        self._lock = threading.Lock()
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        try:
            with self._get_connection() as conn:
                # Check if conversations table exists and has the right schema
                cursor = conn.execute("PRAGMA table_info(conversations)")
                columns = [row[1] for row in cursor.fetchall()]
                
                # Create conversations table if it doesn't exist
                if not columns:
                    conn.execute("""
                        CREATE TABLE conversations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            user_input TEXT NOT NULL,
                            assistant_response TEXT NOT NULL,
                            query_type TEXT,
                            use_rag BOOLEAN DEFAULT 0,
                            retrieved_docs TEXT,
                            reasoning TEXT,
                            risk_evaluation TEXT,
                            response_time REAL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                else:
                    # Add missing columns if they don't exist
                    if 'session_id' not in columns:
                        conn.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
                    if 'reasoning' not in columns:
                        conn.execute("ALTER TABLE conversations ADD COLUMN reasoning TEXT")
                    if 'risk_evaluation' not in columns:
                        conn.execute("ALTER TABLE conversations ADD COLUMN risk_evaluation TEXT")
                    if 'retrieved_docs' not in columns:
                        conn.execute("ALTER TABLE conversations ADD COLUMN retrieved_docs TEXT")
                
                # Create feedback table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER,
                        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                        feedback_text TEXT,
                        user_id TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                    )
                """)
                
                # Create analytics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metadata TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance (only if they don't exist)
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)")
                except:
                    pass  # Index might already exist
                    
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
                except:
                    pass
                    
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_conversation ON feedback(conversation_id)")
                except:
                    pass
                    
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_metric ON analytics(metric_name)")
                except:
                    pass
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def log_conversation(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        query_type: str = None,
        use_rag: bool = False,
        retrieved_docs: List[str] = None,
        reasoning: str = None,
        risk_evaluation: str = None,
        response_time: float = None
    ) -> int:
        """Log a conversation to the database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO conversations 
                    (session_id, user_input, assistant_response, query_type, use_rag, 
                     retrieved_docs, reasoning, risk_evaluation, response_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    user_input,
                    assistant_response,
                    query_type,
                    use_rag,
                    json.dumps(retrieved_docs) if retrieved_docs else None,
                    reasoning,
                    risk_evaluation,
                    response_time
                ))
                
                conversation_id = cursor.lastrowid
                conn.commit()
                
                logger.debug(f"📝 Logged conversation {conversation_id} for session {session_id}")
                return conversation_id
                
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
            return None
    
    def get_conversation_history(
        self,
        session_id: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            with self._get_connection() as conn:
                if session_id:
                    cursor = conn.execute("""
                        SELECT * FROM conversations 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                    """, (session_id, limit, offset))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM conversations 
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                    """, (limit, offset))
                
                conversations = []
                for row in cursor.fetchall():
                    conv = dict(row)
                    # Parse JSON fields
                    if conv['retrieved_docs']:
                        try:
                            conv['retrieved_docs'] = json.loads(conv['retrieved_docs'])
                        except:
                            conv['retrieved_docs'] = []
                    conversations.append(conv)
                
                return conversations
                
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def submit_feedback(
        self,
        conversation_id: int,
        rating: int,
        feedback_text: str = None,
        user_id: str = None
    ) -> bool:
        """Submit user feedback for a conversation"""
        try:
            if not (1 <= rating <= 5):
                raise ValueError("Rating must be between 1 and 5")
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO feedback 
                    (conversation_id, rating, feedback_text, user_id)
                    VALUES (?, ?, ?, ?)
                """, (conversation_id, rating, feedback_text, user_id))
                
                conn.commit()
                logger.info(f"📊 Feedback submitted for conversation {conversation_id}: {rating}/5")
                return True
                
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            with self._get_connection() as conn:
                # Overall feedback stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        AVG(rating) as average_rating,
                        MIN(rating) as min_rating,
                        MAX(rating) as max_rating
                    FROM feedback
                """)
                stats = dict(cursor.fetchone())
                
                # Rating distribution
                cursor = conn.execute("""
                    SELECT rating, COUNT(*) as count
                    FROM feedback
                    GROUP BY rating
                    ORDER BY rating
                """)
                rating_dist = {row['rating']: row['count'] for row in cursor.fetchall()}
                stats['rating_distribution'] = rating_dist
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {}
    
    def log_analytics(self, metric_name: str, metric_value: float, metadata: Dict = None):
        """Log analytics metrics"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO analytics (metric_name, metric_value, metadata)
                    VALUES (?, ?, ?)
                """, (metric_name, metric_value, json.dumps(metadata) if metadata else None))
                
                conn.commit()
                logger.debug(f"📈 Logged metric: {metric_name} = {metric_value}")
                
        except Exception as e:
            logger.error(f"Error logging analytics: {e}")
    
    def get_analytics(self, metric_name: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """Get analytics data"""
        try:
            with self._get_connection() as conn:
                if metric_name:
                    cursor = conn.execute("""
                        SELECT * FROM analytics 
                        WHERE metric_name = ? 
                        AND timestamp >= datetime('now', '-{} days')
                        ORDER BY timestamp DESC
                    """.format(days), (metric_name,))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM analytics 
                        WHERE timestamp >= datetime('now', '-{} days')
                        ORDER BY timestamp DESC
                    """.format(days))
                
                analytics = []
                for row in cursor.fetchall():
                    analytic = dict(row)
                    if analytic['metadata']:
                        try:
                            analytic['metadata'] = json.loads(analytic['metadata'])
                        except:
                            analytic['metadata'] = {}
                    analytics.append(analytic)
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return []
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Conversation stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_conversations,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        AVG(response_time) as avg_response_time,
                        COUNT(CASE WHEN use_rag = 1 THEN 1 END) as rag_conversations
                    FROM conversations
                """)
                conv_stats = dict(cursor.fetchone())
                stats.update(conv_stats)
                
                # Query type distribution
                cursor = conn.execute("""
                    SELECT query_type, COUNT(*) as count
                    FROM conversations
                    WHERE query_type IS NOT NULL
                    GROUP BY query_type
                    ORDER BY count DESC
                """)
                query_types = {row['query_type']: row['count'] for row in cursor.fetchall()}
                stats['query_type_distribution'] = query_types
                
                # Recent activity (last 24 hours)
                cursor = conn.execute("""
                    SELECT COUNT(*) as recent_conversations
                    FROM conversations
                    WHERE timestamp >= datetime('now', '-1 day')
                """)
                recent = dict(cursor.fetchone())
                stats.update(recent)
                
                # Add feedback stats
                feedback_stats = self.get_feedback_stats()
                stats['feedback'] = feedback_stats
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """Clean up old data to keep database size manageable"""
        try:
            with self._get_connection() as conn:
                # Delete old conversations
                cursor = conn.execute("""
                    DELETE FROM conversations 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                deleted_conversations = cursor.rowcount
                
                # Delete old analytics
                cursor = conn.execute("""
                    DELETE FROM analytics 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days_to_keep))
                deleted_analytics = cursor.rowcount
                
                conn.commit()
                
                total_deleted = deleted_conversations + deleted_analytics
                logger.info(f"🧹 Cleaned up {total_deleted} old records")
                return total_deleted
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        try:
            with self._get_connection() as conn:
                # Database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                # Table sizes
                cursor = conn.execute("""
                    SELECT name, COUNT(*) as count
                    FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    GROUP BY name
                """)
                table_sizes = {row['name']: row['count'] for row in cursor.fetchall()}
                
                return {
                    "database_path": self.db_path,
                    "database_size_mb": round(db_size / (1024 * 1024), 2),
                    "table_sizes": table_sizes,
                    "is_accessible": True
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {
                "database_path": self.db_path,
                "database_size_mb": 0,
                "table_sizes": {},
                "is_accessible": False,
                "error": str(e)
            }

# Global database service instance
database_service = DatabaseService()



