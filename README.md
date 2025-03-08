# h2o-gbm
Simple H2O template in Java and Python for GBM

### GBM

	$ export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
	$ mvn clean install 
	$ cp="target/lib/*:target/h2o-gbm-1.0-SNAPSHOT.jar"
	$ java -cp $cp --add-opens java.base/java.lang=ALL-UNNAMED com.mannetroll.analysis.GBMTrainApp
	
	$ export MAVEN_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED"
	$ mvn exec:java

