#include <iostream>
#include <queue>
#include <ctime>
using namespace std;
#pragma warning(disable:4996)
#define PRINTF_ANSWER 0
#define Ans1 1
const int MAX = 16;		//16个数
const int N = 4;		//4阶
const int gsum = 34;	//行列对角线34
const int maxx = 100;
//DFS
int cnt = 0;			//结果的数目
int arr[N + 1][N + 1];	//4*4的数组
int f[MAX + 1];			//标记第几个数的flag
//BFS
typedef struct node
{
	int a[17] = { 0 };
	int used[17] = { 0 };
	int tt = 2;
	//AStar
	int f, g, h;
	//g应该是已经填了的这一行数字的和
	//h应该是没填的数列的（最大值）
	bool operator < (const node& r) const{
		return f < r.f;
	}
}node;
node s;
void DFS_init() {
	memset(arr, 0, sizeof(arr));
	memset(f, 0, sizeof(f));
	f[1] = 1;	arr[1][1] = 1;	cnt = 0;
}
void DFS(int x) {		//第x个数
	int i, j, k, s;
	if (x > 16) {		//都填满了 
		i = arr[1][1] + arr[2][2] + arr[3][3] + arr[4][4];//对角线
		j = arr[1][4] + arr[2][3] + arr[3][2] + arr[4][1];//对角线
		if (i != gsum || j != gsum)		//对角线结果不为34就返回
			return;
		for (i = 1; i <= N; i++) {		//列的和不是34就返回
			s = 0;
			for (j = 1; j <= N; j++)
				s += arr[j][i];
			if (s != gsum)
				return;
		}
		cnt++;
#if PRINTF_ANSWER
		printf("第%d次结果：\n",cnt);
		for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++)
				printf("%d ", arr[i][j]);
			printf("\n");
		}
		printf("\n");
#endif // PRINTF_ANSWER
		return;
	}
	//没填满
	for (i = 2; i <= MAX; i++) {
		if (!f[i]) {	//这个数字i没填
			for (j = 1; j <= N; j++) {
				s = 0;
				for (k = 1; k <= N; k++) {
					if (!arr[j][k]) {	//如果[j,k]这个地方没数字
						f[i] = 1;		//标记数字i已填
						arr[j][k] = i;	
						break;
					}
					s += arr[j][k];		//计算该行的和
				}
				if (f[i])				//1~16都填完了
					break;
				if (s != gsum)			//这一行的和不是34
					return;
			}
#if Ans1
			//只找第一个解
			if (x == 16) {		//都填满了
				i = arr[1][1] + arr[2][2] + arr[3][3] + arr[4][4];//对角线
				j = arr[1][4] + arr[2][3] + arr[3][2] + arr[4][1];//对角线
				if (i != gsum || j != gsum)		//对角线结果不为34就返回
					return;
				for (i = 1; i <= N; i++) {		//列的和不是34就返回
					s = 0;
					for (j = 1; j <= N; j++)
						s += arr[j][i];
					if (s != gsum)
						return;
				}
				break;
			}
#endif // Ans1
			DFS(x + 1);					//下一个
			arr[j][k] = 0;	f[i] = 0;	//回溯
		}
	}
	return;
}

bool Check_Right(int a[]) {
	if (
		(a[1] + a[2] + a[3] + a[4]) != gsum ||
		(a[5] + a[6] + a[7] + a[8]) != gsum ||
		(a[9] + a[10] + a[11] + a[12]) != gsum ||
		(a[13] + a[14] + a[15] + a[16]) != gsum ||
		(a[1] + a[6] + a[11] + a[16]) != gsum ||	//对角线
		(a[4] + a[7] + a[10] + a[13]) != gsum ||	//对角线
		(a[1] + a[5] + a[9] + a[13]) != gsum ||
		(a[2] + a[6] + a[10] + a[14]) != gsum ||
		(a[3] + a[7] + a[11] + a[15]) != gsum ||
		(a[4] + a[8] + a[12] + a[16]) != gsum
		)
		return false;
	return true;
}

void BFS() {

	queue<node> q;
	s.a[1] = 1;
	s.used[1] = 1;
	q.push(s);
	while (!q.empty()) {				//有保留的可用状态
		node st = q.front();
		q.pop();
		if (st.tt == 17) { 
#if Ans1
			break;
#endif // Ans1
			cnt++;
#if PRINTF_ANSWER
			printf("第%d次结果：\n", cnt);
			for (int i = 1; i <= 16; i++) {
				printf("%d ", st.a[i]);
				if (i % 4 == 0)
					printf("\n");
			}
			printf("\n");
#endif // PRINTF_ANSWER
			
		}
		for (int d = 2; d <= 16; d++) {	//16个数
			if (st.used[d] == 0) {		//没用过其中的1个数
				st.a[st.tt] = d;	st.used[d] = 1;	st.tt++;

				switch ((st.tt - 1)) {	//蛋疼的剪枝
				case 3:if ((st.a[1] + st.a[2] + st.a[3]) < 34 && (st.a[1] + st.a[2] + st.a[3]) > 17)
						q.push(st);break;
				case 4:if ((st.a[1] + st.a[2] + st.a[3] + st.a[4]) == 34)
						q.push(st);break;
				case 7:if ((st.a[5] + st.a[6] + st.a[7]) < 34 && (st.a[5] + st.a[6] + st.a[7]) > 17)
						q.push(st);break;
				case 8:if ((st.a[5] + st.a[6] + st.a[7] + st.a[8]) == 34)
						q.push(st);break;
				case 9:if ((st.a[1] + st.a[5] + st.a[9]) < 34 && (st.a[1] + st.a[5] + st.a[9]) > 17)
						q.push(st);break;
				case 10:if ((st.a[2] + st.a[6] + st.a[10]) < 34 && (st.a[2] + st.a[6] + st.a[10]) > 17 &&
						(st.a[4] + st.a[7] + st.a[10]) < 34 && (st.a[4] + st.a[7] + st.a[10]) > 17)
						q.push(st);break;
				case 11:if ((st.a[3] + st.a[7] + st.a[11]) < 34 && (st.a[3] + st.a[7] + st.a[11]) > 17 &&
						(st.a[1] + st.a[6] + st.a[11]) < 34 && (st.a[1] + st.a[6] + st.a[11]) > 17 &&
						(st.a[9] + st.a[10] + st.a[11]) < 34 && (st.a[9] + st.a[10] + st.a[11]) > 17)
						q.push(st);break;
				case 12:if ((st.a[4] + st.a[8] + st.a[12]) < 34 && (st.a[4] + st.a[8] + st.a[12]) > 17 &&
						(st.a[1] + st.a[6] + st.a[11]) < 34 && (st.a[1] + st.a[6] + st.a[11]) > 17 &&
						(st.a[9] + st.a[10] + st.a[11] + st.a[12]) == 34 && (st.a[5] + st.a[8] + st.a[9] + st.a[12]) == 34)
						q.push(st);break;
				case 13:if ((st.a[1] + st.a[5] + st.a[9] + st.a[13]) == 34 &&
						(st.a[4] + st.a[7] + st.a[10] + st.a[13]) == 34)
						q.push(st);break;
				case 14:if ((st.a[2] + st.a[6] + st.a[10] + st.a[14]) == 34)
						q.push(st);break;
				case 15:if ((st.a[3] + st.a[7] + st.a[11] + st.a[15]) == 34 && (st.a[2] + st.a[3] + st.a[14] + st.a[15]) == 34 &&
						(st.a[13] + st.a[14] + st.a[15]) < 34 && (st.a[13] + st.a[14] + st.a[15]) > 17)
						q.push(st);break;
				case 16:if ((st.a[4] + st.a[8] + st.a[12] + st.a[16]) == 34 && (st.a[13] + st.a[14] + st.a[15] + st.a[16]) == 34 &&
						(st.a[1] + st.a[6] + st.a[11] + st.a[16]) == 34 && (st.a[1] + st.a[4] + st.a[13] + st.a[16]) == 34)
						q.push(st);break;
				default:
					q.push(st); break;
				}
				st.tt--;
				st.used[d] = 0;
			}
		}
	}
}

void AStar() {
	priority_queue<node> q;
	s.a[1] = 1;
	s.used[1] = 1;
	q.push(s);
	while (!q.empty()) {				//有保留的可用状态
		node st = q.top();
		q.pop();
		if (st.tt == 17) {
#if Ans1
			break;
#endif // Ans1

			cnt++;
#if PRINTF_ANSWER
			printf("第%d次结果：\n", cnt);
			for (int i = 1; i <= 16; i++) {
				printf("%d ", st.a[i]);
				if (i % 4 == 0)
					printf("\n");
			}
			printf("\n");
#endif // PRINTF_ANSWER
		}
		for (int d = 2; d <= 16; d++) {	//16个数
			if (st.used[d] == 0) {		//没用过其中的1个数
				st.a[st.tt] = d;	st.used[d] = 1;	st.tt++;
				for (int o = 2; o <= 16; o++) {
					if (st.used[o] == 0) {
						st.h = o;
						break;
					}
				}
				for (int i = 0; i <= 4; i++) {//st.tt - 1是当前格
					if (i == 0) {
						st.g = 0;
						continue;
					}
					st.g += st.a[((st.tt - 1 - 1) / 4) * 4 + i];
				}
				st.f = st.g + st.h;
				switch ((st.tt - 1)) {	//蛋疼的剪枝
				case 3:if ((st.a[1] + st.a[2] + st.a[3]) < 34 && (st.a[1] + st.a[2] + st.a[3]) > 17)
					q.push(st); break;
				case 4:if ((st.a[1] + st.a[2] + st.a[3] + st.a[4]) == 34)
					q.push(st); break;
				case 7:if ((st.a[5] + st.a[6] + st.a[7]) < 34 && (st.a[5] + st.a[6] + st.a[7]) > 17)
					q.push(st); break;
				case 8:if ((st.a[5] + st.a[6] + st.a[7] + st.a[8]) == 34)
					q.push(st); break;
				case 9:if ((st.a[1] + st.a[5] + st.a[9]) < 34 && (st.a[1] + st.a[5] + st.a[9]) > 17)
					q.push(st); break;
				case 10:if ((st.a[2] + st.a[6] + st.a[10]) < 34 && (st.a[2] + st.a[6] + st.a[10]) > 17 &&
					(st.a[4] + st.a[7] + st.a[10]) < 34 && (st.a[4] + st.a[7] + st.a[10]) > 17)
					q.push(st); break;
				case 11:if ((st.a[3] + st.a[7] + st.a[11]) < 34 && (st.a[3] + st.a[7] + st.a[11]) > 17 &&
					(st.a[1] + st.a[6] + st.a[11]) < 34 && (st.a[1] + st.a[6] + st.a[11]) > 17 &&
					(st.a[9] + st.a[10] + st.a[11]) < 34 && (st.a[9] + st.a[10] + st.a[11]) > 17)
					q.push(st); break;
				case 12:if ((st.a[4] + st.a[8] + st.a[12]) < 34 && (st.a[4] + st.a[8] + st.a[12]) > 17 &&
					(st.a[1] + st.a[6] + st.a[11]) < 34 && (st.a[1] + st.a[6] + st.a[11]) > 17 &&
					(st.a[9] + st.a[10] + st.a[11] + st.a[12]) == 34 && (st.a[5] + st.a[8] + st.a[9] + st.a[12]) == 34)
					q.push(st); break;
				case 13:if ((st.a[1] + st.a[5] + st.a[9] + st.a[13]) == 34 &&
					(st.a[4] + st.a[7] + st.a[10] + st.a[13]) == 34)
					q.push(st); break;
				case 14:if ((st.a[2] + st.a[6] + st.a[10] + st.a[14]) == 34)
					q.push(st); break;
				case 15:if ((st.a[3] + st.a[7] + st.a[11] + st.a[15]) == 34 && (st.a[2] + st.a[3] + st.a[14] + st.a[15]) == 34 &&
					(st.a[13] + st.a[14] + st.a[15]) < 34 && (st.a[13] + st.a[14] + st.a[15]) > 17)
					q.push(st); break;
				case 16:if ((st.a[4] + st.a[8] + st.a[12] + st.a[16]) == 34 && (st.a[13] + st.a[14] + st.a[15] + st.a[16]) == 34 &&
					(st.a[1] + st.a[6] + st.a[11] + st.a[16]) == 34 && (st.a[1] + st.a[4] + st.a[13] + st.a[16]) == 34)
					q.push(st); break;
				default:
					q.push(st); break;
				}
				st.tt--;
				st.used[d] = 0;
			}
		}
	}
}


int main() {
	
#if PRINTF_ANSWER
	freopen("416Time.txt", "w", stdout);
#endif // PRINTF_ANSWER

	clock_t startTime, endTime;
///*
	startTime = clock();//计时开始
	DFS_init(); DFS(2);
	endTime = clock();//计时结束4
	cout << "DFS time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//*/
///*
	startTime = clock();//计时开始
	BFS();
	endTime = clock();//计时结束
	cout << "BFS time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//*/
///*
	startTime = clock();//计时开始
	AStar();
	endTime = clock();//计时结束
	cout << "AStar time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//*/

#if PRINTF_ANSWER
	fclose(stdout);
#endif // PRINTF_ANSWER
	return 0;
}
