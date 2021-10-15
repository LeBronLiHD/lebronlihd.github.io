---
layout: post
title:  "CIMC New Features Details"
date:   2021-10-15 22:57:09 +0800
category: InterestsPosts
---


# CIMC新增功能明细


## ChessAI

### 残局功能拓展与规则完善

- 在比赛的时候，评委们提出了让我们增加残局、棋谱的功能期望，于是赛后我们进行了象棋AI的功能拓展

#### 1. 棋谱（done）

- 加入棋谱：韬略元机、湖涯集、深渊海阔、梦入神机、梅花谱、桔中秘、锦秘豹略一共7个棋谱，具体详见文件夹 `./srategies/棋谱`

#### 2. 残局

- 用户通过软件自定义残局：用户通过软件自定义每一个棋子的位置，但是如果用户自定义棋子的位置不合法则需要重新自定义
- 视觉驱动自定义残局功能：用户通过直接摆放实物棋子的位置自定义残局，在开始下棋之前，通过机器视觉识别出每一个棋子的类型与坐标，生成残局棋盘

#### 3. 规则完善（done）

- 在象棋中，将和帅如果处于同一行上，两个棋子之间的必须存在其他棋子，否则导致这一局面的一方直接失败。现在该功能已完善（这个可以不说，不是很重要）

### 决策算法提升

- 不能只有一个单线程运行的`alpha-beta`剪枝算法了

#### 1. 动态层数的`alpha-beta`剪枝算法（done）

- 在棋局进行至末期的时候，由于合法的步数有限，可以在保证不延长计算时间的前提下适当增加搜索层数

#### 2. 多线程计算与GPU计算

- 多线程：目前仅仅是单线程计算，计算速度较慢，最多可以使用12线程计算
- GPU：安装`kuda`后使用GPU代替CPU进行计算
- 多说一句：由于`alpha-beta`剪枝算法的时间按复杂度是指数级的，大概$ O(N) = 40^N $，采用多线程计算与GPU计算个人认为最多提升一层，效果不明显

#### 3. 新的决策算法

- 网上查找更加厉害的决策算法，在理解的基础上替换成我们的接口
- 具体详见`./srategies/ChessAI_plan.md`
- 但是不要局限于上面那个文件

## AGV Robot

### 硬件

- 3D打印吸盘升降的舵机驱动零件
- 3D打印主控板、电源模块、散热模块、MPU9050芯片、摄像头、雷达探测的固定零件，组装小车
- 使用SolidWorks图完成AGV小车的转配图

### 软件

#### 1. 保证基本实现

- 在主控板上编写移动、气泵等程序
  - process - 1
    - layer - 1
      - `def init()`
      - `def transfer_gogogo_time(x1. x2)`
      - `def transfer_turn_time(left_or_right)`
      - `def get_current_position()`
      - `def chenge_position(x)`
    - layer - 2
      - `def motor_gogogo(forward_or_backward, duty_cycle, time)`
      - `def turn_direction(left_or_right)`
      - `def gripper_move(up_or_down)`
      - `def suction_cup(open_or_shut)`
    - layer - 3
      - `def move_to_point(x)`
      - `def get_or_drop_chess(get_or_drop)`
  - process - 2
    - `def init_MPU905()`
    - `def read_MPU9050()`

#### 2. 丰富拓展

- 在主控板上摄像头、雷达探测的相应程序
  - 摄像头：判断小车底部是否存在棋子
  - 雷达探测：地形探测
- 在主控板上编写路径规划、控制（比如反馈、PID）算法

## APPENDIX

- 源码：
  - ChessAI: https://github.com/LeBronLiHD/chessRobotSimulation_QT
  - Vision: https://github.com/LeBronLiHD/SRTP_CNN_ChessBoardGenerating

- 接口说明：

  - `singleGame.h`

  - ```c++
    singleGame();	// 没用
    virtual void setLevel(int level);	// 没用
    virtual int getLevel();	// 没用
    virtual void generateBlackAllPossibleMoves();	// 没用
    virtual void displayBlackAllPossibleMoves();	// 没用
    virtual void SdisplayBlackAllPossibleMoves();	// 没用
    virtual void generateRedAllPossibleMoves();	// 没用
    virtual void displayRedAllPossibleMoves();	// 没用
    virtual void SdisplayRedAllPossibleMoves();	// 没用
    virtual void tranStructToClass();	// 没用
    virtual void testChessing(int maxCount = 5);	// 没用
    virtual void testFakeChessing(int maxCount = 5);	// 没用
    virtual void testStepClass();	// 没用
    virtual void testFakeBackMove();	// 没用
    virtual int generateRandomNumber(int maxInt);	// 没用
    virtual bool compareSteps(chessStep last, chessStep current);	// 没用
    
    // real play
    virtual void oneLevelChessing(int maxCount);	// 没用
    virtual void oneLevelChessing_HumanVSAI(int maxCount);	// 没用
    virtual void S_oneLevelChessing(int maxCount);	// 没用
    virtual int oneLevelStepIndex(bool redOrBlack);	// 没用
    virtual int S_oneLevelStepIndex(bool redOrBlack);	// 没用
    virtual void twoLevelChessing(int maxCount);	// 没用
    virtual void twoLevelChessing_HumanVSAI(int maxCount);	// 没用
    virtual void S_twoLevelChessing(int maxCount);	// 没用
    virtual int twoLevelStepIndex(bool redOrBlack);	// 没用
    virtual int S_twoLevelStepIndex(bool redOrBlack);	// 没用
    virtual void threeLevelChessing(int maxCount);	// 没用
    virtual void threeLevelChessing_HumanVSAI(int maxCount);	// 没用
    virtual int threeLevelStepIndex(bool redOrBlack);	// 没用
    virtual void S_threeLevelChessing(int maxCount);	// 没用
    virtual int S_threeLevelStepIndex(bool redOrBlack);	// 没用
    
    virtual void normalPlay(int maxCount);	// 有用
    virtual void normalPlay_HumanVSAI(int maxCount);	// 有用
    virtual void normalPlay_HumanVSHuman(int maxCount);	// 有用
    virtual void normalPlay_HumanVSAI_CIMC(int maxCount);	// 有用
    virtual void normalPlay_HumanVSAI_CIMC_EndGame(int maxCount);	// 有用
    virtual void normalPlay_EndGame(int maxCount);	// 有用
    virtual void normalPlay_HumanVSHuman_EndGame(int maxCount);	// 有用
    virtual void normalPlay_HumanVSAI_EndGame(int maxCount);	// 有用
    
    virtual int MonteCarloTree_black(int depth);	// 有用，新的没写的Ai算法
    virtual int QuiescentSearch_black(int depth);	// 有用，新的没写的Ai算法
    virtual int Quiescent_alpha_beta_getMin(int depth, int curMin);	// 有用，新的没写的Ai算法，附属于QuiescentSearch_black
    virtual int Quiescent_alpha_beta_getMax(int depth, int curMax);	// 有用，新的没写的Ai算法，附属于QuiescentSearch_black
    virtual int alpha_beta_black(int depth);	// 有用，当前Ai算法
    virtual int alpha_beta_red(int depth);	// 有用，当前Ai算法
    virtual int alpha_beta_getMin(int depth, int curMin);	// 有用，当前Ai算法，附属于alpha_beta_black，alpha_beta_black
    virtual int alpha_beta_getMax(int depth, int curMax);	// 有用，当前Ai算法，附属于alpha_beta_black，alpha_beta_black
    virtual bool isHorseCannonStep_red(const chessStep& curStep);	// 没用
    virtual bool isHorseCannonStep_black(const chessStep& curStep);	// 没用
    virtual int alpha_beta_try(int depth, int alpha, int beta, bool redOrBlack);	// 没用
    virtual int normalPlayIndex_old(bool redOrBlack);	// 没用
    virtual int alpha_beta_old(int depth, int alpha, int beta, bool redOfBlack);	// 没用
    
    // interface
    virtual void realMove(chessStep step);	// 没用
    virtual void fakeMove(chessStep step);	// 没用
    virtual void realBackMove(chessStep step, int lastPosX, int lastPosY);	// 没用
    virtual void fakeBackMove(chessStep step, int lastPosX, int lastPosY);	// 没用
    virtual void realMove(chessStep* step);	// 没用
    virtual void fakeMove(chessStep* step);	// 没用
    virtual void realBackMove(chessStep* step, int lastPosX, int lastPosY);	// 没用
    virtual void fakeBackMove(chessStep* step, int lastPosX, int lastPosY);	// 没用
    virtual bool humanMove();	// 没用
    virtual bool humanMove_black();	// 没用
    virtual bool isHumanStepValid(chessStep step);	// 没用
    virtual bool isHumanStepValid_black(chessStep step);	// 没用
    
    virtual int currentSearchDepth_black();	// 没用
    virtual int currentSearchDepth_red();	// 没用
    virtual void changeSearchDepth_black(int currentSearchDepth);	// 没用
    virtual void changeSearchDepth_red(int currentSearchDepth);	// 没用
    virtual void updateSearchDepth_black();	// 没用
    virtual void updateSearchDepth_red();	// 没用
    virtual int currentSearchDepth();	// 没用
    virtual void changeSearchDepth(int currentSearchDepth, QString camp);	// 没用
    
    virtual void deleteStepList(QVector<chessStep*>& stepList);	// 没用
    
    int VisionHumanStepIndex(const QVector<chessStep>& curStepList);	// 没用
    ```