import mysql.connector as mysql_c

class sql_db():
    def __init__(self, addr, db, login, pw = None):
        try:
            self.connection = mysql_c.connect(
                host = addr,
                database = db,
                user = login,
                password = pw
                )
            self.cursor = self.connection.cursor()
            print('Sucssefully connected to MySQLdatabase {}.'.format(db))
        except mysql_c.Error as error:
            print('Failed to connect to MySQL database {} due to {}'.format(db, error))
        
    def create_table(self):
        msg = """CREATE TABLE online_input (
	            age_of_driver int NOT NULL,
                gender int NOT NULL,
                marital_status int NOT NULL,
                safty_rating int NOT NULL,
                annual_income int NOT NULL, high_education_ind int NOT NULL, address_change_ind int NOT NULL,
                living_status int NOT NULL,
                accident_site int NOT NULL, 
                past_num_of_claims int NOT NULL,
                witness_present_ind int NOT NULL,
                liab_prct int NOT NULL,
                channel int NOT NULL,
                policy_report_filed_ind int NOT NULL,
                claim_est_payout int NOT NULL,
                age_of_vehicle int NOT NULL,
                vehicle_category int NOT NULL, vehicle_price int NOT NULL,
                vehicle_color int NOT NULL, vehicle_weight int NOT NULL, fraud int NOT NULL)"""
        try:
            command = msg
            self.cursor.execute(command)
            self.connection.commit()
            print('Sucessfully created table.')
        except mysql_c.Error as error:
            print('Failed to create table in MySQL due to {}'.format(error))

    def insert_query(self, parameters):
        msg = """INSERT INTO online_input (
		age_of_driver,
        gender, 
        marital_status, 
        safty_rating, 
        annual_income,
		high_education_ind, 
        address_change_ind, 
        living_status, 
        accident_site,
		past_num_of_claims, 
        witness_present_ind, 
        liab_prct, 
        channel,
		policy_report_filed_ind, 
        claim_est_payout, 
        age_of_vehicle, 
        vehicle_category,
		vehicle_price, 
        vehicle_color, 
        vehicle_weight, 
        fraud) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s) """
        try:
            command = msg
            record = parameters
            self.cursor.execute(command, record)
            self.connection.commit()
            print('Sucessfully insert query.')
        except mysql_c.Error as error:
            print('Failed to insert into MySQL due to {}'.format(error))

    def close_connection(self):
        if self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print('MySQL connection is closed.')
        else:
            print('MySQL connection is already closed.')

if __name__ == '__main__':
    mysql = sql_db('35.192.15.39', 'predicted', 'root', '1234abc')
    #mysql.create_table()
    parameters = [35, 1, 1, 80, 100000, 1, 0, 1, 1, 0, 1, 50, 1, 1, 10000, 5, 1, 50000, 1, 5000, 0]
    mysql.insert_query(parameters)
    
