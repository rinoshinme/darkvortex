#ifndef DARKVORTEX_LIST_H
#define DARKVORTEX_LIST_H

#include <vector>

template<typename T>
struct Node
{
	T val;
	Node<T>* next;
	Node<T>* prev;
public:
	Node() : next(NULL), prev(NULL) {}
};

template<typename T>
struct List
{
private:
	int size;
	Node<T>* front;
	Node<T>* back;

public:
	List() : size(0), front(NULL), back(NULL) { }
	List(const List<T>& copy) : size(0), front(NULL), back(NULL)
	{
		Node<T>* n = copy.front;
		while (n)
		{
			insert(n->val);
			n = n->next;
		}
	}

	Node<T>* front_node() const { return front; }

	~List() { freeNode(this->front); }

	List<T> operator=(const List<T>& copy)
	{
		freeNode(this->front);
		size = 0;
		back = NULL;
		front = NULL;
		Node<T>* n = copy.front;
		while (n)
		{
			insert(n->val);
			n = n->next;
		}
		return *this;
	}

	void insert(T val)
	{
		Node<T>* new_node = new Node<T>;
		new_node->val = val;
		new_node->next = NULL;
		if (back == NULL)
		{
			front = new_node;
			new_node->prev = NULL;
		}
		else
		{
			back->next = new_node;
			new_node->prev = back;
		}
		back = new_node;
		size += 1;
	}

	void* pop()
	{
		if (back == NULL)
			return 0;
		Node* b = back;
		T val = b->val;
		back = b->prev;
		if (back)
			back->next = NULL;
		delete b;
		size -= 1;
		return (void*)val;
	}

	std::vector<T> toArray() const
	{
		std::vector<T> res;
		Node<T>* n = front;
		while (n)
		{
			res.push_back(n->val);
			n = n->next;
		}
		return res;
	}

private:
	void freeNode(Node<T>* n)
	{
		Node<T>* next;
		while (n)
		{
			next = n->next;
			delete n;
			n = next;
		}
	}
};

#endif
