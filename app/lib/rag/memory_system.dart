import 'dart:convert';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../inference/rwkv_engine.dart';

/// RAG 记忆系统
/// 本地向量检索 + SQLite 存储
class RAGMemorySystem {
  static final RAGMemorySystem _instance = RAGMemorySystem._internal();
  factory RAGMemorySystem() => _instance;
  RAGMemorySystem._internal();
  
  Database? _db;
  final RWKVInferenceEngine _engine = RWKVInferenceEngine();
  
  /// 初始化数据库
  Future<void> initialize() async {
    if (_db != null) return;
    
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'pet_memory.db');
    
    _db = await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        // 记忆表
        await db.execute('''
          CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            importance REAL NOT NULL,
            created_at INTEGER NOT NULL,
            access_count INTEGER DEFAULT 0,
            last_accessed INTEGER
          )
        ''');
        
        // 对话历史表
        await db.execute('''
          CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            memory_id INTEGER,
            FOREIGN KEY (memory_id) REFERENCES memories (id)
          )
        ''');
        
        // 创建索引
        await db.execute('''
          CREATE INDEX idx_memories_importance ON memories(importance DESC)
        ''');
        
        await db.execute('''
          CREATE INDEX idx_conversations_timestamp ON conversations(timestamp DESC)
        ''');
      },
    );
  }
  
  /// 添加记忆
  Future<int> addMemory({
    required String content,
    required double importance,
  }) async {
    if (_db == null) {
      throw Exception('Memory system not initialized');
    }
    
    // 获取嵌入向量
    final embedding = await _engine.getEmbedding(content);
    final embeddingStr = embedding.join(',');
    
    final now = DateTime.now().millisecondsSinceEpoch;
    
    return await _db!.insert('memories', {
      'content': content,
      'embedding': embeddingStr,
      'importance': importance,
      'created_at': now,
      'access_count': 0,
    });
  }
  
  /// 添加对话
  Future<void> addConversation({
    required String role,
    required String content,
    int? memoryId,
  }) async {
    if (_db == null) {
      throw Exception('Memory system not initialized');
    }
    
    final now = DateTime.now().millisecondsSinceEpoch;
    
    await _db!.insert('conversations', {
      'role': role,
      'content': content,
      'timestamp': now,
      'memory_id': memoryId,
    });
  }
  
  /// 检索相关记忆
  Future<List<Map<String, dynamic>>> searchMemories({
    required String query,
    int limit = 5,
    double minImportance = 0.0,
  }) async {
    if (_db == null) {
      throw Exception('Memory system not initialized');
    }
    
    // 获取查询嵌入
    final queryEmbedding = await _engine.getEmbedding(query);
    
    // 获取所有记忆
    final memories = await _db!.query(
      'memories',
      where: 'importance >= ?',
      whereArgs: [minImportance],
      orderBy: 'importance DESC',
    );
    
    // 计算相似度并排序
    final scored = memories.map((m) {
      final embedding = (m['embedding'] as String)
          .split(',')
          .map((s) => double.parse(s))
          .toList();
      
      final similarity = _cosineSimilarity(queryEmbedding, embedding);
      
      return {
        ...m,
        'similarity': similarity,
      };
    }).toList();
    
    // 按相似度排序
    scored.sort((a, b) => (b['similarity'] as double).compareTo(a['similarity'] as double));
    
    // 更新访问计数
    final topMemories = scored.take(limit).toList();
    for (final m in topMemories) {
      await _db!.update(
        'memories',
        {
          'access_count': (m['access_count'] as int) + 1,
          'last_accessed': DateTime.now().millisecondsSinceEpoch,
        },
        where: 'id = ?',
        whereArgs: [m['id']],
      );
    }
    
    return topMemories;
  }
  
  /// 获取对话历史
  Future<List<Map<String, dynamic>>> getConversationHistory({
    int limit = 20,
    int? beforeTimestamp,
  }) async {
    if (_db == null) {
      throw Exception('Memory system not initialized');
    }
    
    return await _db!.query(
      'conversations',
      where: beforeTimestamp != null ? 'timestamp < ?' : null,
      whereArgs: beforeTimestamp != null ? [beforeTimestamp] : null,
      orderBy: 'timestamp DESC',
      limit: limit,
    );
  }
  
  /// 清理低重要性记忆
  Future<void> cleanupOldMemories({
    double threshold = 0.3,
    int maxAgeDays = 30,
  }) async {
    if (_db == null) {
      throw Exception('Memory system not initialized');
    }
    
    final cutoff = DateTime.now()
        .subtract(Duration(days: maxAgeDays))
        .millisecondsSinceEpoch;
    
    await _db!.delete(
      'memories',
      where: 'importance < ? AND created_at < ?',
      whereArgs: [threshold, cutoff],
    );
  }
  
  /// 计算余弦相似度
  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) return 0.0;
    
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    if (normA == 0 || normB == 0) return 0.0;
    
    return dotProduct / (normA * normB);
  }
  
  /// 导出记忆数据
  Future<String> exportMemories() async {
    if (_db == null) {
      throw Exception('Memory system not initialized');
    }
    
    final memories = await _db!.query('memories');
    final conversations = await _db!.query('conversations');
    
    final data = {
      'memories': memories,
      'conversations': conversations,
      'exported_at': DateTime.now().toIso8601String(),
    };
    
    return jsonEncode(data);
  }
  
  /// 关闭数据库
  Future<void> close() async {
    await _db?.close();
    _db = null;
  }
}
