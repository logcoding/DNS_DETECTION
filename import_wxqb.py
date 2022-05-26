import json
import pymysql
import os

conn = pymysql.connect(
    host = '172.16.118.245',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'test_db',
    charset = 'utf8mb4'
)
cur = conn.cursor()

createTableSql1 = """
                create table if not exists test_db.wxqb(
                job_type TEXT default null, 
                create_time TEXT default null, 
                group_name TEXT default null, 
                source_type TEXT default null, 
                content_en TEXT default null, 
                title TEXT default null, 
                content TEXT default null, 
                site_name TEXT default null, 
                product_type TEXT default null, 
                job_name TEXT default null, 
                site_url TEXT default null, 
                publish_time TEXT default null, 
                title_en TEXT default null, 
                id TEXT default null, 
                pageurl TEXT default null
                )DEFAULT CHARSET=utf8mb4;
            """
createTableSql2 = """
                create table if not exists test_db.wxqb_kafka_news(
                UUID TEXT default null, 
                AddTime TEXT default null, 
                Nation_s TEXT default null, 
                id TEXT default null, 
                send_time TEXT default null, 
                create_time TEXT default null, 
                publish_time TEXT default null, 
                job_type TEXT default null, 
                group_name TEXT default null, 
                source_type TEXT default null, 
                content_en TEXT default null, 
                title TEXT default null, 
                content TEXT default null, 
                content_html TEXT default null, 
                site_name TEXT default null, 
                product_type TEXT default null, 
                job_name TEXT default null, 
                site_url TEXT default null, 
                pageurl TEXT default null, 
                title_en TEXT default null, 
                src_ip TEXT default null, 
                poster_id TEXT default null, 
                poster_name TEXT default null, 
                modify_time TEXT default null, 
                section_name TEXT default null, 
                section_id TEXT default null, 
                post_id TEXT default null, 
                post_parentid TEXT default null, 
                parent_postername TEXT default null, 
                parent_postid TEXT default null, 
                post_type TEXT default null, 
                data_source TEXT default null, 
                top_domain TEXT default null, 
                content_length TEXT default null, 
                device TEXT default null, 
                comment_num TEXT default null, 
                forward_num TEXT default null, 
                read_num TEXT default null, 
                like_num TEXT default null, 
                mentioned_posterids TEXT default null, 
                floor_num TEXT default null, 
                score TEXT default null, 
                import_type TEXT default null, 
                language_code TEXT default null, 
                edit_time TEXT default null, 
                picurls TEXT default null, 
                group_ids TEXT default null, 
                group_names TEXT default null, 
                file_urls TEXT default null, 
                file_types TEXT default null, 
                abstract2 TEXT default null, 
                alarm_topicid TEXT default null, 
                level_oneclasstype TEXT default null, 
                sentiment_type TEXT default null, 
                sentiment_value TEXT default null, 
                message_summary TEXT default null, 
                guang_yiclasstype TEXT default null, 
                enterprise_id TEXT default null, 
                user_topicid TEXT default null, 
                geo_name TEXT default null, 
                word_list TEXT default null, 
                is_rubbish TEXT default null, 
                title_wordslist TEXT default null, 
                content_wordslist TEXT default null, 
                title_keyword TEXT default null, 
                keyword TEXT default null, 
                organization TEXT default null, 
                person TEXT default null, 
                location_cluster TEXT default null, 
                subject TEXT default null, 
                is_cream INT(10) default null, 
                consum_time TEXT default null, 
                is_send INT(10) default null , 
                SysFlag TEXT default null
                )DEFAULT CHARSET=utf8mb4;
                """

json_path = r'E:\caowc\wxqb_data'
for root,_,files in os.walk(json_path):
    for file in files:
        # if file.endswith('txt'):
        with open(os.path.join(root,file),'r',encoding='utf_8_sig') as f:
            for line in f.readlines():
                dic = json.loads(line)
                keys = ','.join(dic.keys())
                valueslist = [dicv for dicv in dic.values()]
                valuesTuple = tuple(valueslist)
                values = ', '.join(['%s']*len(dic))
                if file.endswith('txt'):
                    table = 'test_db.wxqb_kafka_news'
                    insertSql = 'INSERT INTO {table}({keys}) VALUES ({values})'.format(table=table,keys=keys,values=values)
                    # conn.ping()
                    cur.execute(createTableSql2)
                    cur.execute(insertSql,valuesTuple)
                    conn.commit()
                else:
                    table = 'test_db.wxqb'
                    insertSql = 'INSERT INTO {table}({keys}) VALUES ({values})'.format(table=table, keys=keys,
                                                                                       values=values)
                    cur.execute(createTableSql1)
                    cur.execute(insertSql, valuesTuple)
                    conn.commit()
    conn.close()



