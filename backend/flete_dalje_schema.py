"""Targeted, transactional migration of delivery numbering to recipient scope."""
import json
import unicodedata


def client_key(name):
    return ' '.join(unicodedata.normalize('NFKC', name).lower().split())


def migrate_client_numbering(engine):
    with engine.begin() as connection:
        # Explicit SQLite transaction keeps table replacement and copy atomic.
        connection.exec_driver_sql('BEGIN IMMEDIATE')
        columns = connection.exec_driver_sql('PRAGMA table_info(flete_dalje_documents)').all()
        if not columns or any(row[1] == 'client_key' for row in columns):
            return
        connection.exec_driver_sql('''CREATE TABLE flete_dalje_documents_scoped (
            id VARCHAR(80) NOT NULL PRIMARY KEY,
            number INTEGER NOT NULL,
            client_key VARCHAR(1000) NOT NULL,
            version INTEGER NOT NULL,
            content_json TEXT NOT NULL,
            CONSTRAINT uq_fd_client_number UNIQUE (client_key, number)
        )''')
        rows = connection.exec_driver_sql('SELECT id, number, version, content_json FROM flete_dalje_documents').mappings().all()
        if rows:
            connection.exec_driver_sql('''INSERT INTO flete_dalje_documents_scoped
                (id, number, client_key, version, content_json) VALUES (?, ?, ?, ?, ?)''',
                [(row['id'], row['number'], client_key(json.loads(row['content_json']).get('recipient', '')),
                  row['version'], row['content_json']) for row in rows])
        connection.exec_driver_sql('DROP TABLE flete_dalje_documents')
        connection.exec_driver_sql('ALTER TABLE flete_dalje_documents_scoped RENAME TO flete_dalje_documents')
