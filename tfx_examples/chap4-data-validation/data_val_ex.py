import tensorflow_data_validation as tfdv

#Importing existing dataset from csv or tfrecord

#stats = tfdv.generate_statistics_from_csv(data_location='../../tmp/heart.csv',delimiter=',')
stats = tfdv.generate_statistics_from_tfrecord(data_location='../../tmp/heart.tfrecords')
stats2 = tfdv.generate_statistics_from_tfrecord(data_location='../../tmp/heart2.tfrecords')
tfdv.visualize_statistics(lhs_statistics=stats,rhs_statistics=stats2,lhs_name="1",rhs_name="2")

schema = tfdv.infer_schema(stats)
anomalies = tfdv.validate_statistics(statistics=stats2,schema=schema)
tfdv.display_anomalies(anomalies)
#Comparing two datasets



"""
#Visualize schema of tfrecord (dataset)
schema = tfdv.infer_schema(stats)
tfdv.display_schema(schema)
"""



#print(stats)
