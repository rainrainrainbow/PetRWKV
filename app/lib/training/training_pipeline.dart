import 'dart:async';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import '../inference/rwkv_engine.dart';
import '../rag/memory_system.dart';
import '../pet/pet_state.dart';

/// 设备端训练管线
/// 负责整理对话历史、构建数据集、执行 LoRA 微调
class TrainingPipeline {
  static final TrainingPipeline _instance = TrainingPipeline._internal();
  factory TrainingPipeline() => _instance;
  TrainingPipeline._internal();
  
  final RWKVInferenceEngine _engine = RWKVInferenceEngine();
  final RAGMemorySystem _memory = RAGMemorySystem();
  
  bool _isTraining = false;
  StreamController<TrainingProgress>? _progressController;
  
  /// 获取训练进度流
  Stream<TrainingProgress> get progressStream {
    _progressController ??= StreamController<TrainingProgress>.broadcast();
    return _progressController!.stream;
  }
  
  /// 是否正在训练
  bool get isTraining => _isTraining;
  
  /// 执行训练任务
  Future<void> runTraining(PetState petState) async {
    if (_isTraining) {
      print('Training already in progress');
      return;
    }
    
    _isTraining = true;
    
    try {
      _reportProgress(TrainingProgress(message: '整理对话历史...', progress: 0.1));
      
      // 1. 获取最近的对话历史
      final conversations = await _memory.getConversationHistory(limit: 100);
      
      if (conversations.isEmpty) {
        _reportProgress(TrainingProgress(message: '没有新的对话数据', progress: 1.0));
        return;
      }
      
      _reportProgress(TrainingProgress(message: '评估对话重要性...', progress: 0.2));
      
      // 2. 评估对话重要性并筛选
      final importantConversations = await _filterImportantConversations(conversations);
      
      if (importantConversations.isEmpty) {
        _reportProgress(TrainingProgress(message: '没有足够重要的对话', progress: 1.0));
        return;
      }
      
      _reportProgress(TrainingProgress(message: '构建训练数据集...', progress: 0.3));
      
      // 3. 构建训练数据集
      final datasetPath = await _buildTrainingDataset(importantConversations);
      
      _reportProgress(TrainingProgress(message: '准备 LoRA 微调...', progress: 0.4));
      
      // 4. 执行 LoRA 微调
      await _performLoraTraining(datasetPath);
      
      _reportProgress(TrainingProgress(message: '更新宠物状态...', progress: 0.9));
      
      // 5. 更新宠物状态
      petState.recordTraining();
      petState.addExperience(importantConversations.length * 10);
      
      // 6. 将重要对话入库 RAG
      for (final conv in importantConversations) {
        await _memory.addMemory(
          content: conv['content'] as String,
          importance: 0.8,
        );
      }
      
      _reportProgress(TrainingProgress(message: '训练完成！', progress: 1.0));
      
    } catch (e) {
      _reportProgress(TrainingProgress(message: '训练失败: $e', progress: 1.0, error: true));
      rethrow;
    } finally {
      _isTraining = false;
    }
  }
  
  /// 筛选重要对话
  Future<List<Map<String, dynamic>>> _filterImportantConversations(
    List<Map<String, dynamic>> conversations,
  ) async {
    final important = <Map<String, dynamic>>[];
    
    for (final conv in conversations) {
      final content = conv['content'] as String;
      final role = conv['role'] as String;
      
      // 只处理用户消息
      if (role != 'user') continue;
      
      // 简单的重要性评估
      final importance = _evaluateImportance(content);
      
      if (importance >= 0.6) {
        important.add({...conv, 'importance': importance});
      }
    }
    
    // 按重要性排序
    important.sort((a, b) => (b['importance'] as double).compareTo(a['importance'] as double));
    
    // 取前 50 条
    return important.take(50).toList();
  }
  
  /// 评估对话重要性
  double _evaluateImportance(String content) {
    double score = 0.5;
    
    // 长度因素
    if (content.length > 100) score += 0.1;
    if (content.length > 300) score += 0.1;
    
    // 关键词因素
    final importantKeywords = ['喜欢', '讨厌', '重要', '必须', '记住', '永远', '喜欢', '爱'];
    for (final keyword in importantKeywords) {
      if (content.contains(keyword)) {
        score += 0.05;
      }
    }
    
    // 问题因素
    if (content.contains('?') || content.contains('？')) {
      score += 0.05;
    }
    
    return score.clamp(0.0, 1.0);
  }
  
  /// 构建训练数据集
  Future<String> _buildTrainingDataset(List<Map<String, dynamic>> conversations) async {
    final appDir = await getApplicationDocumentsDirectory();
    final datasetDir = Directory('${appDir.path}/training_data');
    if (!await datasetDir.exists()) {
      await datasetDir.create(recursive: true);
    }
    
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final datasetPath = '${datasetDir.path}/dataset_$timestamp.jsonl';
    
    final file = File(datasetPath);
    final sink = file.openWrite();
    
    try {
      // 构建对话对
      for (int i = 0; i < conversations.length - 1; i++) {
        final userMsg = conversations[i];
        final assistantMsg = conversations[i + 1];
        
        if (userMsg['role'] == 'user' && assistantMsg['role'] == 'assistant') {
          final trainingPair = {
            'instruction': userMsg['content'],
            'output': assistantMsg['content'],
          };
          
          sink.writeln('${trainingPair.toString()}');
        }
      }
    } finally {
      await sink.flush();
      await sink.close();
    }
    
    return datasetPath;
  }
  
  /// 执行 LoRA 微调
  Future<void> _performLoraTraining(String datasetPath) async {
    // 调用 C++ 训练引擎
    // 这里需要通过 FFI 调用 C++ 训练函数
    // 实际实现需要在 C++ 层完成
    
    // 模拟训练过程
    for (int i = 0; i <= 10; i++) {
      await Future.delayed(Duration(seconds: 1));
      _reportProgress(TrainingProgress(
        message: '训练中... ${i * 10}%',
        progress: 0.4 + (i * 0.05),
      ));
    }
  }
  
  /// 报告训练进度
  void _reportProgress(TrainingProgress progress) {
    _progressController?.add(progress);
  }
  
  /// 释放资源
  void dispose() {
    _progressController?.close();
  }
}

/// 训练进度
class TrainingProgress {
  final String message;
  final double progress;
  final bool error;
  
  TrainingProgress({
    required this.message,
    required this.progress,
    this.error = false,
  });
}
