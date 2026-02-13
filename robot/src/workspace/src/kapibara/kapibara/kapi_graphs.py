import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import uuid
import numpy as np

import sqlite3

class ConnectionDatabase:
    def __init__(self, db_name="my_database.db"):
        """Initialize the connection and create the table if it doesn't exist."""
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """Creates a table."""
        query = """
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            _from TEXT NOT NULL,
            _to TEXT NOT NULL,
            linear FLOAT NOT NULL,
            angular FLOAT NOT NULL
        )
        """
        self.cursor.execute(query)
        self.connection.commit()

    def add_connection(self, _from, _to, linear, angular):
        """Add a new connection"""
        query = ( "INSERT INTO entries (_from, _to, linear, angular) "
                    "VALUES (?, ?, ?, ?)" )
        self.cursor.execute(query, (_from, _to, linear, angular))

    def get_connections_from(self,_from):
        """Get connections from specific node"""
        query = "SELECT * FROM entries WHERE _from = ?"
        self.cursor.execute(query, (_from,))
        return self.cursor.fetchall()
    
    def get_connections_to(self,_to):
        """Get connections to specific node"""
        query = "SELECT * FROM entries WHERE _to = ?"
        self.cursor.execute(query, (_to,))
        return self.cursor.fetchall()
    
    def delete_connections_from(self, _from):
        """Deletes all records that comes from specific node"""
        query = "DELETE FROM entries WHERE _from = ?"
        self.cursor.execute(query, (_from,))
        self.connection.commit()

    def get_all_entries(self):
        """Retrieves all records from the table."""
        self.cursor.execute("SELECT * FROM entries")
        return self.cursor.fetchall()

    def close(self):
        """Closes the database connection."""
        self.connection.close()
        print("Database connection closed.")
    
    def push(self):
        """Synchronize database"""
        self.connection.commit()

class Graph:
    def __init__(self,db_name:str):
        self.node_db = chromadb.PersistentClient(
            path=f"{db_name}_node",
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
            )
        self.conn_db = ConnectionDatabase(f"{db_name}_conn")
        self.nodes = self.node_db.create_collection("nodes",get_or_create=True)
    
    def update_nodes_metadata(self,ids:list[str],scores:list[float]):
        self.nodes.upsert(
            ids=ids,
            metadatas=[
                {
                    "score":score
                } 
                for score in scores
                       ]
        )
        
    def add_node(self,embedding:np.ndarray,score:float,point:tuple=None):
        
        res = self.nodes.query(query_embeddings=[embedding],n_results=1)
                        
        if len(res['distances'][0]) > 0 and res['distances'][0][0] < 0.01:
            id = res['ids'][0][0]
        else:
            id = str(uuid.uuid4())
        
        self.nodes.upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[{
                "score":score
                }]
        )
        
        return id
    
    def connect_nodes(self,_from:str,_to:str,actions:tuple[float,float]):
        self.conn_db.add_connection(_from,_to,actions[0],actions[1])
        
    def get_node(self,embedding:np.ndarray):
        res = self.nodes.query(query_embeddings=[embedding],n_results=1)
        
        if len(res['distances'][0]) > 0 and res['distances'][0][0] < 0.01:
            return (res['ids'][0][0],res['metadatas'][0][0]['score'])
        else:
            return None
        
    def get_node_by_ids(self,ids:list[str]):
        res = self.nodes.get(ids=ids)
        
        if len(res["ids"]) == 0:
            return None
        
        out = []
        
        for i in enumerate(res[ids]):
            out.append(
                (res['ids'][i][0],res['metadatas'][i][0]['score'])
            )
        
        return out
                
    def remove_node(self,id:str):
        self.nodes.delete([id])
        self.conn_db.delete_connections_from(id)
        
    def get_connections(self,id:str):
        res = []
        
        connections = self.conn_db.get_connections_from(id)
        
        for conn in connections:
            res.append(
                (conn[2],conn[3],conn[4])
            )
        
        return res
    
    def get_backwards_connections(self,id:str):
        res = []
        
        connections = self.conn_db.get_connections_to(id)
        
        for conn in connections:
            res.append(
                (conn[1],conn[3],conn[4])
            )
        
        return res
    
    def sync(self):
        self.conn_db.push()