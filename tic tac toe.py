'''
    校准包括：
    1、校准偏差
    2、校准十字箭头
    3、校准机械臂夹取
'''
import cv2   #导入库
import numpy as np
import time
import threading
import replace
import os
import random

import z_uart as myUart
import z_beep as myBeep
import z_kinematics as kms
import tkinter as tk
from math import *

import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
save_estimation_1=dict();
class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):#np.nditer按照从左到右，从上到下的顺序遍历数组值
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val #对于每种状态，都有一个唯一的hashvalue与其对应

    # check whether a player has won the game, or it's a tie
    def is_end(self):#检查游戏是否结束
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):#根据坐标放棋子
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state




def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)#在（i,j）处放一个current_symbol棋子
                new_hash = new_state.hash()#计算新状态的hash值
                if new_hash not in all_states:#如果是新的状态
                    is_end = new_state.is_end()#判断是否分了胜负
                    all_states[new_hash] = (new_state, is_end)#all_states 是一个字典，这一步是在字典中添加值
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)#递归算法，直到分出胜负平

def get_all_states():
    current_symbol = 1#初始化为1代表玩家先手下棋
    current_state = State()
    all_states = dict()#all_states是一个字典型变量
    all_states[current_state.hash()] = (current_state, current_state.is_end())#往字典中存值
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states

# all possible board configurations
all_states = get_all_states()#all_states是一个存了所有可能的形式的一个字典值

class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        self.p1 = player1 #见第325行，player1和player2实际上是两个类
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)#给当前的状态打分
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1#yield相当于return，但是下次调用时就直接执行下一行
            yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()#切换棋手
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            i, j, symbol = player.act()#这里的symbol是在judger初始化的地方确定的
            # act方法选择最最有可能胜利的点
            next_state_hash = current_state.next_state(i, j, symbol).hash()#在上一行算出的i,j位置放一个棋子，并计算哈希值
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()#绘制出棋盘
            if is_end:
                return current_state.winner


# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)
        #print(self.greedy)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states:
            state, is_end = all_states[hash_val]#state是当前的状态，is_end是胜负判断的结果
            if is_end:
                if state.winner == self.symbol:#赢家和symbol一样
                    self.estimations[hash_val] = 1.0 #这个状态和胜负结果存到字典中，胜利给1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5 #平局给0.5
                else:
                    self.estimations[hash_val] = 0 #失败给0
            else:
                self.estimations[hash_val] = 0.5 #未结束给0.5

    # update value estimation
    def backup(self):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (
                    self.estimations[states[i + 1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon: # 0.1的可能性会随机下，避免出现不可能出现的情况
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)#按照列表中的第一个值降序排列，就是取最有可能获胜的
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)
            return self.estimations

# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // BOARD_COLS
        j = data % BOARD_COLS
        return i, j, self.symbol


def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judger.reset()

    player1.save_policy()
    player2.save_policy()
    print(player2.estimations)
    print(player1.estimations)
    return player2.estimations



def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")



def next_state_hash(data,i,j):
    new_data=data.copy()
    new_data[i,j]=-1;
    hash_value=0;
    for k in np.nditer(new_data):  # np.nditer按照从左到右，从上到下的顺序遍历数组值
        hash_value = hash_value * 3 + k + 1
    return hash_value  # 对于每种状态，都有一个唯一的hashvalue与其对应


def determine_the_next_step(data):
    #state = self.states[-1]
    next_states = []
    next_positions = []
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if data[i, j] == 0:
                next_positions.append([i, j])
                next_states.append(next_state_hash(data,i,j))
    #print(next_states)
    #print(next_positions)
    #
    values = []

    for hash_val, pos in zip(next_states, next_positions):
        values.append((save_estimation_1[hash_val], pos))

    # to select one of the actions of equal value at random due to Python's sort is stable
    print(values)
    np.random.shuffle(values)
    values.sort(key=lambda x: x[0], reverse=True)  # 按照列表中的第一个值降序排列，就是取最有可能获胜的
    action = values[0][1]
    #action.append(self.symbol)
    return action

#关于色域范围可以百度 HSV
#百科：https://baike.baidu.com/item/HSV/547122?fr=aladdin
#参考：https://blog.csdn.net/leo_888/article/details/88284251

# 要识别的颜色字典
color_dist = {
             #'red':   {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
             'blue':  {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
             };

row_number = 3
column_number = 3
maximum_x = 0
maximum_y = 0
minimum_x = 0
minimum_y = 0
status_of_game = 0
index_1 = -1
axis_x_cur = 0
axis_y_cur = 0
axis_x_lst = 0
axis_y_lst = 0
blue_block_count =0
judge_random = 0
judge_win = 0
axis_block=[]
c_x_blue_block =0
c_y_blue_block =0
game_level =0
win_counter=0;
draw_counter=0;
lose_counter=0;
robotic_counter=0

axis_block.append([42,102,70])
axis_block.append([2,100,70])
axis_block.append([-42,100,70])
axis_block.append([43,62,70])
axis_block.append([0,61,70])
axis_block.append([-40,62,70])
axis_block.append([45,12,70])
axis_block.append([2,16,70])
axis_block.append([-38,16,70])
selct_blue_position = []

#使用之前 关闭mjpg进程 ps -elf|grep mjpg  找到进程号，杀死进程  sudo kill -9 xxx   xxx代表进程号
#cat /dev/video

img_w = 320
img_h = 240

cap = cv2.VideoCapture(-1)  #打开摄像头 最大范围 640×480
cap.set(3,img_w)  #设置画面宽度
cap.set(4,img_h)  #设置画面高度

def check_win(lst_1 = [], lst_2 = [] ,lst_3 = []):#检查游戏是否结束
    lst = []
    lst.append(lst_1)
    lst.append(lst_2)
    lst.append(lst_3)
    lst=np.matrix(lst)
    results=[];   
    judge_win = 0
    for i in range(3):
        results.append(np.sum(lst[i,:]));
            
    for i in range(3):
        results.append(np.sum(lst[:,i]));
    
    diagonal_1 = 0
    diagonal_2 = 0
    for i in range(3):
        diagonal_1 += lst[i,i];
        diagonal_2 += lst[i,2-i];
    results.append(diagonal_1);
    results.append(diagonal_2);
    
    for result in results:
        if result == 3:
            judge_win = 1;
        if result == -3:
            judge_win = -1;
    #print("check_win result="result)
          
    return judge_win


#逆运动学算法
def kinematics_move(x,y,z,mytime): 
    global servo_angle,servo_pwm
    if y < 0 :
        return 0;
    #寻找最佳角度
    flag = 0;
    my_min = 0
    for i in range(-135,0) :
        if 0 == kms.kinematics_analysis(x,y,z,i):
            if(i<my_min):
                my_min = i;
            flag = 1;
        
    #用3号舵机与水平最大的夹角作为最佳值
    if(flag) :
        kms.kinematics_analysis(x,y,z,my_min);
        testStr = '{'
        for j in range(0,4) :
            #set_servo(j, servo_pwm[j], time);
            #print(servo_pwm[j])
            testStr += "#%03dP%04dT%04d!" % (j,kms.servo_pwm[j],mytime)
        testStr += '}'
        print(testStr)
        myUart.uart_send_str(testStr)
        return 1;
    
    return 0;


#初始x 初始y 初始z 目标x 目标y 目标z 云台角度 抓取时爪子的角度
def start_carry(s_x,s_y,s_z,d_x,d_y,d_z,servo_yuntai,servo_zhuazi):
    
    #myUart.uart_send_str('#255P1500T2000!')
    #time.sleep(1)
    
    #第一步 运动到复位位置
    kinematics_move(0,100,200,1000)
    print("111")
    time.sleep(1)
     
    #第二步 张开爪子 旋转爪子
    testStr = "#%03dP%04dT%04d!#%03dP%04dT%04d!#005P1200T0300!" % (0, servo_yuntai, 1000,4, servo_zhuazi, 1000)
    myUart.uart_send_str(testStr)
    print("222")
    time.sleep(1)
        
    #第三步 运动到识别位置
    kinematics_move(s_x,s_y,s_z,1000)
    print("333")
    time.sleep(2)
    
    #第四步 闭合爪子
    myUart.uart_send_str('#005P1800T1000!#005P1800T1000!')
    print("444")
    time.sleep(1)
    
    #第五步 抬起臂
    #myUart.uart_send_str('#001P2000T2000!#004P1500T2000!')
    if d_y == 0:
        angle2 = 0;
    testStr = "#001P1500T2000!#%03dP%04dT%04d!" % (4, 1500, 1000)
    print(testStr)
    myUart.uart_send_str(testStr)
    print("555")
    time.sleep(2)
        
    #第六步 运行到放置位置
    kinematics_move(d_x,d_y,d_z,2000)
    time.sleep(3)
    kinematics_move(d_x,d_y,d_z-30,2000)
    print("666")
    time.sleep(3)
    
    #第七步 张开爪子
    myUart.uart_send_str('#005P1600T0300!#005P1600T0300!')
    print("777")
    time.sleep(0.3)
    
    #第八步 抬起臂
    myUart.uart_send_str('#001P1500T1000!#004P1500T1000!')
    print("888")
    time.sleep(2)
    
    #第九步 运动到复位位置
    myUart.uart_send_str('#255P1500T1000!#255P1500T1000!')
    print("999")
    time.sleep(2)

#程序执行处
myBeep.setup_beep()
myUart.setup_uart(115200)
kms.setup_kinematics(106,90,75,200)#初始化

#发出哔哔哔作为开机声音
myBeep.beep(0.1)
myBeep.beep(0.1)
myBeep.beep(0.1)

lastTime = time.time();

myUart.uart_send_str('#255P1500T1000!#255P1500T1000!')
time.sleep(2)
kinematics_move(0,100,200,1000)
print("111")
time.sleep(1)
final_ordered_list = []
#无限循环
while 1: #进入无线循环
    #将摄像头拍摄到的画面作为frame的值
    
    ret,frame = cap.read()

#     ret, thresh = cv2.threshold(mask,127,255,0)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
## 阈值分割
    ret,thresh = cv2.threshold(gray,127,255,1)

    ## 对二值图像执行膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(19,19))     
    dilated = cv2.dilate(thresh,kernel)

    ## 轮廓提取，cv2.RETR_TREE表示建立层级结构
#   image, contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    ## 提取小方格，其父轮廓都为0号轮廓
    
    boxes = []
    indexs = []
    number_boxes = []
    row_0 = []
    row_1 = []
    row_2 = []
    ordered_list_0 = []
    ordered_list_1 = []
    ordered_list_2 = []
    

    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == 0:
            boxes.append(hierarchy[0][i])
            #print(hierarchy[0][i])
            indexs.append(i)
    for j in range(len(boxes)):
        if boxes[j][2] == -1 : #方格中空白
            cnt=1
            x,y,w,h = cv2.boundingRect(contours[indexs[j]])
            if boxes[j][2] ==-1 and w >30 :
                number_boxes.append([cnt,x,y,w,h])
                cnt = cnt +1
                img = cv2.rectangle(frame,(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)
                cv2.imshow('img',img)
            
    if number_boxes:       
        for k in range(1,4):
            x = number_boxes[k-1][1]
            y = number_boxes[k-1][2]
            w = number_boxes[k-1][3]
            h = number_boxes[k-1][4]
            row_0.append([x,y,w,h])
        ordered_list_0 = sorted(row_0, key=lambda k: k[0])
        
        for k in range(0,3):
            #print(ordered_list_0[k])
            x = ordered_list_0[k][0]
            y = ordered_list_0[k][1]
            w = ordered_list_0[k][2]
            h = ordered_list_0[k][3]
            final_ordered_list.append([2,k,x,y,w,h])
           
        for k in range(4,7):
            x = number_boxes[k-1][1]
            y = number_boxes[k-1][2]
            w = number_boxes[k-1][3]
            h = number_boxes[k-1][4]
            row_1.append([x,y,w,h])
        ordered_list_1 = sorted(row_1,key=lambda k: k[0])
        
        for k in range(0,3):
            #print(ordered_list_1[k])
            x = ordered_list_1[k][0]
            y = ordered_list_1[k][1]
            w = ordered_list_1[k][2]
            h = ordered_list_1[k][3]
            final_ordered_list.append([1,k,x,y,w,h])
           
        
        for k in range(7,10):
            x = number_boxes[k-1][1]
            y = number_boxes[k-1][2]
            w = number_boxes[k-1][3]
            h = number_boxes[k-1][4]
            row_2.append([x,y,w,h])
        ordered_list_2 = sorted(row_2,key=lambda k: k[0])
        for k in range(0,3):
            #print(ordered_list_2[k])
            x = ordered_list_2[k][0]
            y = ordered_list_2[k][1]
            w = ordered_list_2[k][2]
            h = ordered_list_2[k][3]
            final_ordered_list.append([0,k,x,y,w,h])
        for k in range (0,9):
            print(final_ordered_list[k])
        print("\n")
    

  
    if cv2.waitKey(5) & 0xFF == ord(' '): #如果按空格就退出循环
        break
time.sleep(2)
cap.release()
cv2.destroyAllWindows()
#读取棋盘结束

easy_counter=0
medium_counter=0
hard_counter=0

window = tk.Tk()
window.title('select the level of the game')
window.geometry("500x300+150+150")
#window["background"] = "blue"

var = tk.StringVar()
l_0 =tk.Label(window,bg="yellow",width=30,text='Please choose your level')
l_0.pack()
l_1 = tk.Label(window,width = 30,text='')
l_1.pack()

def print_selection():
    if var.get() == '1':
        l_1.config(text='You have selected easy level')
    elif var.get() == '2':
        l_1.config(text='You have selected medium level')
    elif var.get() == '3':
        l_1.config(text='You have selected hard level')
    
r1 = tk.Radiobutton(window,text='Easy',variable = var,value ='1',command=print_selection)
r1.place(x=150,y=50)

r2 = tk.Radiobutton(window,text='Medium',variable = var,value ='2',command=print_selection)
r2.place(x=150,y=100)

r3 = tk.Radiobutton(window,text='Hard',variable = var,value ='3',command=print_selection)
r3.place(x=150,y=150)
Exit = tk.Button(text = 'Comfirm and get started!',command = window.destroy)
Exit.place(x=150,y=200)
window.mainloop()
print("the value of var is",var.get())
#定位棋子
cap = cv2.VideoCapture(-1)  #打开摄像头 最大范围 640×480
cap.set(3,img_w)  #设置画面宽度
cap.set(4,img_h)  #设置画面高度
ret,image=cap.read()


easy_count=0
medium_count=0
hard_count=0
game_flag=1
counter_cnt=0
judge_draw=0
while game_flag==1:
    if var.get()=='1' or var.get()=='2' or var.get()=='3':
        while 1:            
            ret,img_2 = cap.read()
            hsv = cv2.cvtColor(img_2,cv2.COLOR_BGR2HSV)
            grid=np.zeros((3,3))      
            Cp = [] #Center position
            counter_cnt = 0
            c_x_blue_block_final =0
            c_y_blue_block_final =0
            select_blue_position = []
            recognition_blue_position = []
            for k in range(0,9):
                row_number = final_ordered_list[k][0]
                column_number = final_ordered_list[k][1]
                x = final_ordered_list[k][2]
                y = final_ordered_list[k][3]
                w = final_ordered_list[k][4]
                h = final_ordered_list[k][5]
                centerColor = hsv[round((2*y+h)/2),round((2*x+w)/2)]
                print(centerColor)
                
                if (170 < centerColor[0]<180 and  110< centerColor[1] < 256 and 60 < centerColor[2] < 256):
                    grid[row_number][column_number]=1
                elif (-1< centerColor[0]<10 and  100< centerColor[1] < 256 and 60 < centerColor[2] < 256):
                    grid[row_number][column_number]=1
                elif (100< centerColor[0]<150 and 0< centerColor[1]< 256 and 20< centerColor[2]< 230):
                    grid[row_number][column_number]=-1
                if (grid[row_number][column_number]!=0):
                    counter_cnt = counter_cnt+1
                    
            print(grid[0])
            print(grid[1])
            print(grid[2])
            print('\n')
            print('cnt=',counter_cnt)
            time.sleep(2)
            next_step = []
            counter_diagonal_1 = 0
            counter_diagonal_2 = 0
            counter_diagonal_3 = 0
            counter_diagonal_4 = 0
               
            judge_win = check_win(grid[0],grid[1],grid[2])
            if counter_cnt==9 and judge_win==0:
                pass
            elif counter_cnt%2 == 1 and var.get()=='2' and judge_win==0:#medium level
                medium_counter=medium_counter+1
                if judge_win == 0:
                    for i in range(0,3):
                        counter = 0
                        for k in range (0,3):
                            counter=np.sum(grid[i][k]);
                        if counter ==-2:
                            for k in range(0,3):
                                if grid[i][k] == 0:
                                    index_1 =-3*i+k+6
                                    print("the next step should be (i,k): (",i,",",k,")")
                                    print("\n")
                                    status_of_game = 1#连成三子直接胜利
                       
                            
                    for k in range(0,3):
                        counter = 0
                        for i in range(0,3):
                            counter=np.sum(grid[i][k]);
                        if counter == -2:
                            for i in range(0,3):
                                if grid[i][k] == 0:
                                    index_1 = -3*i+k+6
                                    print("the next step should be (i,k): (",i,",",k,")")
                                    print("\n")
                                    status_of_game = 1
                        
                                                            
                    for i in range(0,3):
                        counter_diagonal_1=np.sum(grid[i][2-i]);
                        if counter_diagonal_1 == -2:
                            for i in range(0,3):
                                k = 2-i
                                if grid[i][k] == 0:
                                    print("the next step should be (i,k): (",i,",",2-i,")")
                                    print("\n")
                                    index_1 = -3*i+k+6
                                    status_of_game = 1                             
                                               
                    for i in range(0,3):
                        counter_diagonal_2=np.sum(grid[i][i]);
                        if counter_diagonal_2 == -2:
                            for i in range(0,3):
                                if grid[i][i]  == 0:
                                    print("the next step should be (i,k): (",i,",",i,")")
                                    print("\n")
                                    index_1 = -3*i+k+6
                                    status_of_game = 1
                                            
                    if status_of_game == 0:#不能直接连成3个，则防止对手直接胜利
                        for i in range(0,3):
                            counter = 0
                            for k in range (0,3):
                                if grid[i][k] ==1:
                                    counter = counter + 1
                                if counter ==3:
                                    judge_win = 1
                                if counter ==2:
                                    for k in range(0,3):
                                        if grid[i][k] == 0:
                                            index_1 =-3*i+k+6
                                            print("the next step should be (i,k): (",i,",",k,")")
                                            print("\n")
                        
                        for k in range(0,3):
                            counter = 0
                            for i in range(0,3):
                                if grid[i][k] == 1:
                                    counter = counter+1
                                if counter == 2:
                                    for i in range(0,3):
                                        if grid[i][k] == 0:
                                            index_1 = -3*i+k+6
                                            print("the next step should be (i,k): (",i,",",k,")")
                                            print("\n")
                              
                                            
                        for i in range(0,3):
                            k = 2-i 
                            if grid[i][k]==1:
                                counter_diagonal_3 = counter_diagonal_3 +1       
                            if counter_diagonal_3 == 2:
                                for i in range(0,3):
                                    k = 2-i
                                    if grid[i][k] == 0:
                                        print("the next step should be (i,k): (",i,",",k,")")
                                        print("\n")
                                        index_1 = -3*i+k+6
                                    
                        for i in range(0,3):
                            k = i
                            if grid[i][k]==1:
                                counter_diagonal_4 = counter_diagonal_4 +1
                            if counter_diagonal_4 == 2:
                                for i in range(0,3):
                                    k = i
                                    if grid[i][k]  == 0:
                                        print("the next step should be (i,k): (",i,",",k,")")
                                        print("\n")
                                        index_1 = -3*i+k+6
                            
                        if index_1 ==-1:            
                            remained_grid =[0,1,2,3,4,5,6,7,8]
                            for i in range(0,3):                
                                for k in range(0,3):
                                    if grid[i][k]==-1 or grid[i][k] ==1:
                                        index_step = -3*i+k+6
                                        remained_grid.remove(index_step)
                            print(remained_grid)
                            index_1 = random.choice(remained_grid)
                            judge_random = 1
                            i = 2 - int (index_1/3)
                            k = index_1 % 3
                            print("the next step should be (i,k): (",i,",",k,")")
                elif judge_win == 1 or judge_win == -1:
                    pass
            elif counter_cnt%2 ==1 and var.get() =='1' and judge_win==0: #easy level
                easy_counter=easy_counter+1;
                judge_win = check_win(grid[0],grid[1],grid[2])
                if judge_win ==0:
                    remained_grid_1 =[0,1,2,3,4,5,6,7,8]
                    for i in range(0,3):
                        for k in range(0,3):
                            if grid[i][k]==-1 or grid[i][k]==1:
                                index_step = -3*i+k+6
                                remained_grid_1.remove(index_step)
                    index_1 = random.choice(remained_grid_1 )
                    i = 2 -int(index_1/3)
                    k = index_1 % 3
                    judge_random = 1
                    print("the next step should be (i,k): (",i,",",k,")")
                    (judge_win)
                elif judge_win == 1 or judge_win == -1:
                    pass
            elif counter_cnt%2 ==1 and var.get() =='3' and judge_win==0: #hard level
                hard_counter=hard_counter+1;
                judge_win = check_win(grid[0],grid[1],grid[2])
                #save_estimation_1=train(int(1e6))
                with open('policy_second.bin','rb')as f:
                    save_estimation_1=pickle.load(f)
        #compete(int(10e4))
                check_points=0
                for i in range(0,3):
                    for k in range(0,3):
                        check_points=check_points+grid[i][k]
                        
                if check_points!=1:
                    index_1=random.choice([0,1,2,3,4,5,6,7,8])
                else:
                    a=np.mat([grid[0],
                              grid[1],
                              grid[2]]);
                    print(determine_the_next_step(a));
                    i=determine_the_next_step(a)[0]
                    k=determine_the_next_step(a)[1]
                    index_1=index_1 = -3*i+k+6;
                    print(index_1)
                    print("the code is executed!")
            elif judge_win==1 or judge_win==-1:
                pass
                                    
            elif judge_win == 0 and counter_cnt%2 == 1:
            
                for i in color_dist:
                #1-高斯滤波GaussianBlur() 让图片模糊
                    frame = cv2.GaussianBlur(img_2,(5,5),0)
            
                #2-转换HSV的样式 以便检测
                    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
                
                #3-查找字典
                    mask = cv2.inRange(hsv, color_dist[i]['Lower'], color_dist[i]['Upper'])
                
                #4-腐蚀图像
                    mask = cv2.erode(mask,None,iterations=2)
                
                #高斯模糊
                    mask = cv2.GaussianBlur(mask,(3,3),0)
                
                #图像合并
                    res = cv2.bitwise_and(frame,frame,mask=mask)       
                #6-边缘检测
                    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] 
                
                 
                #if len(cnts) >0 : #通过边缘检测来确定所识别物体的位置信息得到相对坐标
                for k in range(0,9):           
                    if maximum_x < final_ordered_list[k][2]+final_ordered_list[k][4]:
                        maximum_x = final_ordered_list[k][2]+final_ordered_list[k][4]
                    if maximum_y < final_ordered_list[k][3]+final_ordered_list[k][5]:
                        maximum_y = final_ordered_list[k][3]+final_ordered_list[k][5]
                for k in range(0,9):
                    if minimum_x > final_ordered_list[k][2]:
                        minimum_x = final_ordered_list[k][2]
                    if minimum_y > final_ordered_list[k][3]:
                        minimum_y = final_ordered_list[k][3]
                print("maximum_x=%d,maximum_y=%d,minimum_x=%d,minimum_y=%d"%(maximum_x,maximum_y,minimum_x,minimum_y))    
                for i in range(len(cnts)):
                    select_blue_position.append(cnts[i])
                    #print("cnt[",i,"]=",cnts[i])
                    #cnt = max(cnts,key=cv2.contourArea)
                for i in range(len(select_blue_position)):
                    
                    rect = cv2.minAreaRect(select_blue_position[i])
                    c_x_blue_block, c_y_blue_block = rect[0]
                    if minimum_x < c_x_blue_block < maximum_x and minimum_y < c_y_blue_block < maximum_y:
                        pass
                        #select_blue_position.remove(select_blue_block[i])
                    else:
                        recognition_blue_position.append(select_blue_position[i])
                        
                    # 获取最小外接矩形的4个顶点
                    #box = cv2.boxPoints(rect)
                 #获取坐标 长宽 角度
                    #c_angle = rect[2]
                    #c_x_blue_block, c_y_blue_block = rect[0]
                for k in range (len(recognition_blue_position)):
                    cnt = max(recognition_blue_position,key=cv2.contourArea)
                    rectangular_shape = cv2.minAreaRect(cnt)
                    c_angle = rectangular_shape[2]
                    c_x_blue_block_final ,c_y_blue_block_final = rectangular_shape[0]
             
              
                (index_1)
                c_x_arm = int(-0.5952*c_x_blue_block_final + 68.258)
                c_y_arm = int(0.5647*c_y_blue_block_final - 2.6619)
                c_x = 160
                c_y = 120
                c_x_camera = final_ordered_list[8-index_1][2]
                c_y_camera = final_ordered_list[8-index_1][3]
                print(c_x_camera,c_y_camera)
                if c_x_camera !=0 and c_y_camera!=0:
                    axis_x_cur = c_x_camera
                    axis_y_cur = c_y_camera
                if abs(axis_x_cur-axis_x_lst) < 5 and abs(axis_y_cur - axis_y_lst) < 5:
                    blue_block_count = blue_block_count +1
                elif judge_random ==1 :
                    blue_block_count = blue_block_count +1
                else:
                    blue_block_count = 0
                print("blue_block_count is ",blue_block_count)
                print("axis_x_cur",axis_x_cur)
                print("axis_y_cur",axis_y_cur)
                print("axis_x_lst",axis_x_lst)
                print("axis_y_lst",axis_y_lst)
                axis_x_lst = c_x_camera
                axis_y_lst = c_y_camera
                if blue_block_count >1 and counter_cnt==2*robotic_counter+1:
                    blue_block_count =0
                    print("the code is exercuted")          
                    robotic_arm_move =1
                    c_x = 0+(c_x-160)
                    c_y = 180-(c_y-120)
                    
                     
                    if(c_x == 0 and c_y != 0) :
                        theta6 = 0.0;
                    elif(c_y == 0 and c_x > 0):
                        theta6 = 90;
                    elif(c_y == 0 and c_x < 0):
                        theta6 = -90;
                    else :
                        theta6 = atan(c_x/c_y)*180.0/pi;
                    
                    theta6 = -theta6
                    
                    #计算云台旋转角度
                    angle_yuntai = atan(c_x/c_y)*180.0/pi;
                    servo_yuntai = (int)(1500-2000.0 * angle_yuntai/ 270.0);
                    print('servo_yuntai:', angle_yuntai)
                                     
                    #计算爪子旋转角度
                    if(c_angle<0):
                        c_angle = c_angle+90
                    #c_angle = 0   
                    if (c_angle>45):
                        c_angle = 90-c_angle
        #
                    if(angle_yuntai<0):
                        servo_zhuazi = (int)(1500-2000.0 * (c_angle-abs(angle_yuntai))/ 270.0);
                        print("------------------------")
                    else:
                        servo_zhuazi = (int)(1500-2000.0 * (-c_angle+abs(angle_yuntai))/ 270.0);
                        print("++++++++++++++++++++++")
        #                                 
                    print('servo_zhuazi:', servo_zhuazi)
        #                             
        #                                                         
                    start_carry(c_x+c_x_arm-10,c_y+c_y_arm,40,c_x+axis_block[index_1][0],c_y+axis_block[index_1][1], axis_block[index_1][2], servo_yuntai, servo_zhuazi);
                    print("the code is executed!")
                    robotic_counter=robotic_counter+1;
                    time.sleep(0.5)
       
        #cv2.imshow('frame',frame) #将具体的测试效果显示出来
        #cv2.imshow('mask',mask)
        #cv2.imshow('res',res)
                
            if counter_cnt == 9 and judge_win==0:
                judge_draw=1
            if judge_win == 1 or judge_win ==-1 or judge_draw == 1 :
                if judge_win==1:
                    win_counter=win_counter+1;
                if judge_win==0:
                    draw_counter=draw_counter+1;
                if judge_win==-1:
                    lose_counter=lose_counter+1;
                myBeep.beep(0.1)
                myBeep.beep(0.1)
                myBeep.beep(0.1)
                myBeep.beep(0.1)
                myBeep.beep(0.1)
                myBeep.beep(0.1)
                
                print("the code is executed!")
                window_1=tk.Tk()
                window_1.title("This the end of the game")
                window_1.geometry("500x300+150+150")
                tag=''
                variable_1 = tk.StringVar()
                if judge_win==1:
                    tag='Congratulations!You have win the game!Want another try?'
                elif judge_win==-1:
                    tag='Sorry,you have lost the game.Want another try?'
                else:
                    tag='It is a draw game. Want another try?'
                label_0=tk.Label(window_1,bg="yellow",width=60,text=tag)
                label_0.pack()
                label_1=tk.Label(window_1,width=60,text='')
                label_1.pack()
                def choose_selection():
                    if variable_1.get()=='1':
                        label_1.config(text="You will have another try,please remove the blocks first");
                    if variable_1.get()=='2':
                        label_1.config(text="You will stop the game");
                rb1=tk.Radiobutton(window_1,text='Yes',variable=variable_1,value='1',command=choose_selection)
                rb1.place(x=150,y=50)
                rb2=tk.Radiobutton(window_1,text='No',variable=variable_1,value='2',command=choose_selection)
                rb2.place(x=150,y=100)
                Exit_button=tk.Button(text="Confirm and exit",command= window_1.destroy)
                Exit_button.place(x=150,y=150)
                window_1.mainloop()
                if variable_1.get()=='2':
                    total_trials=win_counter+draw_counter+lose_counter;
                    print("You have tried %d times\n"  %(total_trials))
                    print("win:%d times\t draw:%d times\t lose:%d times\t" %(win_counter,draw_counter,lose_counter))
                    game_flag=0;
                    break;
                
        if cv2.waitKey(5) & 0xFF == 27: #如果按了ESC就退出 当然也可以自己设置
            break

cap.release()
cv2.destroyAllWindows() #后面两句是常规操作,每次使用摄像头都需要这样设置一波


    