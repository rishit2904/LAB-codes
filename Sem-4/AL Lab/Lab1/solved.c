//write a program to implement stack and queue using circular linked list


#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>

struct node
{
	int info;
	struct node *link;
};

typedef struct node* NODE;

NODE getnode()
{
	NODE x;
	
	x = (NODE) malloc(sizeof(struct node));
	
	if(x == NULL)
	{
		printf("Cannot create node\n");
		return NULL;
	}
	
	return x;
}

void freenode(NODE x)
{
	free(x);
}

NODE push(int item, NODE top)
{
	NODE temp;
	
	temp = getnode();	
	temp->info = item;
	
	if(top == NULL)
	{
		top = temp;
		top->link = top;
		return top;
	}
	else
		temp->link = top->link;
	
	top->link = temp;
	
	return top;
}

NODE pop(NODE top)
{
	NODE first;
	
	if(top == NULL)
	{
		printf("Stack Underflow. Cannot delete.\n");
		return NULL;
	}
	
	if(top->link == top)
	{
		printf("The item deleted is %d\n", top->info);
		freenode(top);
		return NULL;
	}
	
	first = top->link;
	
	printf("Item deleted is %d\n", first->info);
	
	top->link = first->link;
	
	freenode(first);
	
	return top;
}

void display(NODE top)
{
	NODE cur;
	
	if(top == NULL)
	{
		printf("Stack Underflow. Cannot display.\n");
		return;
	}
	
	if(top->link == top)
	{
		printf("%d\n", top->info);
		return;
	}
	
	cur = top;
	
	while(cur->link != top)
	{
		cur = cur->link;
		printf("%d\n", cur->info);
	}
	
	cur = cur->link;
	printf("%d\n", cur->info);
}

void main()
{
	int choice, item;
	
	NODE top;
	
	top = NULL;
	
	printf("Stacks using Circular Linked List\n");
	printf("Insert and Delete to the front of the list\n");
	
	for(;;)
	{
		printf("1. Push\n2. Pop\n3. Display\n4. Exit\n");
		printf("Choice:\t");
		scanf("%d", &choice);
		
		switch(choice)
		{
			case 1:
				printf("Enter the item to be inserted:\t");
				scanf("%d", &item);
				
				top = push(item, top);
				
				break;
			
			case 2:
				top = pop(top);
				
				break;
				
			case 3:
				display(top);
				
				break;
				
			default:
				exit(0);
		}
	}
}
