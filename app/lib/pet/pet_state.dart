import 'package:flutter/foundation.dart';

/// 宠物进化阶段
enum PetStage {
  egg,      // 神秘蛋 - 初始状态
  baby,     // 幼崽 - 完成首次微调
  youth,    // 少年 - 累计7天微调
  adult,    // 成年 - 累计30天微调
  legend,   // 传说 - 累计90天微调
}

/// 宠物个性特征
class PetPersonality {
  final double curiosity;    // 好奇心 0-1
  final double affection;    // 亲密度 0-1
  final double playfulness;  // 活泼度 0-1
  final double wisdom;       // 智慧度 0-1
  
  const PetPersonality({
    this.curiosity = 0.5,
    this.affection = 0.5,
    this.playfulness = 0.5,
    this.wisdom = 0.5,
  });
  
  PetPersonality copyWith({
    double? curiosity,
    double? affection,
    double? playfulness,
    double? wisdom,
  }) {
    return PetPersonality(
      curiosity: curiosity ?? this.curiosity,
      affection: affection ?? this.affection,
      playfulness: playfulness ?? this.playfulness,
      wisdom: wisdom ?? this.wisdom,
    );
  }
  
  Map<String, double> toJson() => {
    'curiosity': curiosity,
    'affection': affection,
    'playfulness': playfulness,
    'wisdom': wisdom,
  };
  
  factory PetPersonality.fromJson(Map<String, dynamic> json) {
    return PetPersonality(
      curiosity: (json['curiosity'] as num?)?.toDouble() ?? 0.5,
      affection: (json['affection'] as num?)?.toDouble() ?? 0.5,
      playfulness: (json['playfulness'] as num?)?.toDouble() ?? 0.5,
      wisdom: (json['wisdom'] as num?)?.toDouble() ?? 0.5,
    );
  }
}

/// 宠物状态管理
class PetState extends ChangeNotifier {
  String _name = '小宠';
  PetStage _stage = PetStage.egg;
  int _level = 0;
  int _experience = 0;
  DateTime _createdAt = DateTime.now();
  DateTime? _lastTrainingAt;
  int _totalTrainingDays = 0;
  PetPersonality _personality = const PetPersonality();
  
  // Getters
  String get name => _name;
  PetStage get stage => _stage;
  int get level => _level;
  int get experience => _experience;
  DateTime get createdAt => _createdAt;
  DateTime? get lastTrainingAt => _lastTrainingAt;
  int get totalTrainingDays => _totalTrainingDays;
  PetPersonality get personality => _personality;
  
  /// 获取当前阶段名称
  String get stageName {
    switch (_stage) {
      case PetStage.egg: return '神秘蛋';
      case PetStage.baby: return '幼崽';
      case PetStage.youth: return '少年';
      case PetStage.adult: return '成年';
      case PetStage.legend: return '传说';
    }
  }
  
  /// 获取下一阶段所需经验
  int get nextStageExp {
    switch (_stage) {
      case PetStage.egg: return 100;
      case PetStage.baby: return 500;
      case PetStage.youth: return 2000;
      case PetStage.adult: return 5000;
      case PetStage.legend: return 99999;
    }
  }
  
  /// 升级经验
  void addExperience(int exp) {
    _experience += exp;
    _checkEvolution();
    notifyListeners();
  }
  
  /// 检查是否进化
  void _checkEvolution() {
    PetStage? newStage;
    
    switch (_stage) {
      case PetStage.egg:
        if (_totalTrainingDays >= 1) newStage = PetStage.baby;
        break;
      case PetStage.baby:
        if (_totalTrainingDays >= 7) newStage = PetStage.youth;
        break;
      case PetStage.youth:
        if (_totalTrainingDays >= 30) newStage = PetStage.adult;
        break;
      case PetStage.adult:
        if (_totalTrainingDays >= 90) newStage = PetStage.legend;
        break;
      case PetStage.legend:
        break;
    }
    
    if (newStage != null && newStage != _stage) {
      _stage = newStage;
      _level++;
    }
  }
  
  /// 记录训练
  void recordTraining() {
    _lastTrainingAt = DateTime.now();
    _totalTrainingDays++;
    _checkEvolution();
    notifyListeners();
  }
  
  /// 更新个性
  void updatePersonality(PetPersonality personality) {
    _personality = personality;
    notifyListeners();
  }
  
  /// 设置名字
  void setName(String name) {
    _name = name;
    notifyListeners();
  }
  
  /// 序列化为 JSON
  Map<String, dynamic> toJson() => {
    'name': _name,
    'stage': _stage.index,
    'level': _level,
    'experience': _experience,
    'createdAt': _createdAt.toIso8601String(),
    'lastTrainingAt': _lastTrainingAt?.toIso8601String(),
    'totalTrainingDays': _totalTrainingDays,
    'personality': _personality.toJson(),
  };
  
  /// 从 JSON 反序列化
  factory PetState.fromJson(Map<String, dynamic> json) {
    final state = PetState();
    state._name = json['name'] as String? ?? '小宠';
    state._stage = PetStage.values[json['stage'] as int? ?? 0];
    state._level = json['level'] as int? ?? 0;
    state._experience = json['experience'] as int? ?? 0;
    state._createdAt = DateTime.parse(json['createdAt'] as String);
    state._lastTrainingAt = json['lastTrainingAt'] != null 
        ? DateTime.parse(json['lastTrainingAt'] as String) 
        : null;
    state._totalTrainingDays = json['totalTrainingDays'] as int? ?? 0;
    state._personality = json['personality'] != null
        ? PetPersonality.fromJson(json['personality'] as Map<String, dynamic>)
        : const PetPersonality();
    return state;
  }
}
