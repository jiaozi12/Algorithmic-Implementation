# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:36:23 2019

@author: Administrator
"""

class Node(object):
    '''单链表节点'''
    def __init__(self,item):
        self.item = item
        self.next = None
        
class LinkList(object):
    '''单链表'''
    def __init__(self):
        self.head = None
    
    def isEmpty(self):
        '''判断链表是否为空'''
        return self.head == None
    
    def length(self):
        '''获取链表长度'''
        count = 0
        cur = self.head
        while cur is not None:
            count += 1
            cur = cur.next
        return count
    
    def search(self,item):
        '''查找链表中是否有与item相等的节点。若有返回所在下标，无返回None'''
        cur = self.head
        count = 0
        while cur is not None:
            if cur.item == item:
                return count
            cur = cur.next
            count += 1
            
    def get_pos(self,pos):
        '''返回指定下标的节点'''
        if type(pos) is not int:
            pos = int(pos)
        if pos < 0:
            #下标小于0，返回首节点
            return self.head.item
        elif pos > self.length()-1:
            #下标大于链表长度，返回尾节点
            cur = self.head
            while cur is not None:
                cur = cur.next
            return cur.item
        else:
            cur = self.head
            count = 0
            while cur is not None:
                if count == pos:
                    return cur.item
                else:
                    cur = cur.next
                    count += 1
                    
    def travel(self):
        '''遍历链表'''
        cur = self.head
        res = []
        while cur is not None:
            res.append(cur.item)
            cur = cur.next
        return res
    
    def insert_ahead(self,item):
        '''在首节点前插入节点'''
        node = Node(item)
        if self.isEmpty():
            self.head = node
        else:
            node.next = self.head
            self.head = node
        
    def insert_brear(self,item):
        '''在尾节点后插入节点'''
        node = Node(item)
        if self.isEmpty():
            #空链表，将head指向node
            self.head = node
        else:
            cur = self.head
            while cur.next is not None:
                #将cur指向尾节点
                cur = cur.next
            cur.next = node
    
    def insert(self,item,pos):
        '''在指定下标位置插入节点'''
        if type(pos) is not int:
            pos = int(pos)
        if pos <= 0:
            #下标小于等于0，在首节点前插入
            self.insert_ahead(item)
        elif pos > (self.length()-1):
            #下标大于链表长度，在尾节点后插入
            self.insert_brear(item)
        else:
            node = Node(item)
            cur = self.head
            count = 0
            while count < pos-1:
                cur = cur.next
                count += 1
            cur_next = cur.next
            cur.next = node
            node.next = cur_next
    
    def remove(self,item):
        '''删除节点'''
        if self.isEmpty():
            return
        else:
            if self.head.item == item:
                #删除首节点
                if self.head.next is None:
                    #链表只有一个节点
                    self.head = None
                else:
                    self.head = self.head.next
            else:
                cur = self.head
                while cur.next is not None:
                    if cur.next.item == item:
                        #cur指向要删除节点的前一个节点
                        cur.next = cur.next.next
                    cur = cur.next

link = LinkList()
print('添加三个元素1  2  3.0')
link.insert_ahead(1)
link.insert_brear(3.0)
link.insert(2,1)

print('便利输出链表：',link.travel())

print('输出链表长度：',link.length())

print('链表是否为空：',link.isEmpty())

print('查询链表中元素2的下标：',link.search(2))

print('查询链表下标为0的元素：',link.get_pos(0))

link.remove(2)
print('删除元素2后的链表长度：',link.length(),'    链表：',link.travel())

link.remove(1)
link.remove(3.0)
print('删除元素1和3.0后，链表是否为空：',link.isEmpty())