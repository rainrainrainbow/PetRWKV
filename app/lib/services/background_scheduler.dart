import 'dart:async';
import 'dart:io';
import 'package:battery_plus/battery_plus.dart';
import 'package:connectivity_plus/connectivity_plus.dart';

/// 后台服务调度器
/// 检测充电状态和闲置状态，触发训练任务
class BackgroundScheduler {
  static final BackgroundScheduler _instance = BackgroundScheduler._internal();
  factory BackgroundScheduler() => _instance;
  BackgroundScheduler._internal();
  
  final Battery _battery = Battery();
  final Connectivity _connectivity = Connectivity();
  
  StreamSubscription? _batterySubscription;
  Timer? _idleCheckTimer;
  
  bool _isCharging = false;
  bool _isIdle = false;
  DateTime? _lastActivityTime;
  
  Function? _onTrainingTrigger;
  
  /// 初始化调度器
  Future<void> initialize({Function? onTrainingTrigger}) async {
    _onTrainingTrigger = onTrainingTrigger;
    _lastActivityTime = DateTime.now();
    
    // 监听电池状态
    _batterySubscription = _battery.onBatteryStateChanged.listen((state) {
      _isCharging = state == BatteryState.charging || state == BatteryState.full;
      _checkTrainingConditions();
    });
    
    // 定期检查闲置状态
    _idleCheckTimer = Timer.periodic(Duration(minutes: 5), (_) {
      _checkIdleStatus();
      _checkTrainingConditions();
    });
  }
  
  /// 记录用户活动
  void recordActivity() {
    _lastActivityTime = DateTime.now();
    _isIdle = false;
  }
  
  /// 检查闲置状态
  void _checkIdleStatus() {
    if (_lastActivityTime == null) return;
    
    final idleDuration = DateTime.now().difference(_lastActivityTime!);
    _isIdle = idleDuration.inMinutes >= 10; // 10分钟无活动视为闲置
  }
  
  /// 检查是否满足训练条件
  void _checkTrainingConditions() {
    // 条件：充电中 或 闲置超过10分钟
    final shouldTrain = _isCharging || _isIdle;
    
    if (shouldTrain && _onTrainingTrigger != null) {
      _onTrainingTrigger!();
    }
  }
  
  /// 获取当前状态
  Map<String, dynamic> getStatus() {
    return {
      'isCharging': _isCharging,
      'isIdle': _isIdle,
      'lastActivity': _lastActivityTime?.toIso8601String(),
    };
  }
  
  /// 释放资源
  void dispose() {
    _batterySubscription?.cancel();
    _idleCheckTimer?.cancel();
  }
}
