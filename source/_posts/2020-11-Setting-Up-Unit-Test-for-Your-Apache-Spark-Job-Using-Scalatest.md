---
title: Setting Up Unit Test for Your Apache Spark Job Using Scalatest
date: 2020-11-28 08:15:10
tags:
---

By nature, machine learning models that run on production need to deal with... well... data, presumably lots of them. There will be times that among many data that our model need to deal with, there will be bad ones. In which case, machine learning models tend to either immediately stop processing data, or continued on with processing and produce smelly result. The impact of both are bad.

Unit testing ML models equip us as developers with an extra confidence to put models in production, by giving a way to an isolated modules in an ML pipeline to face various edge cases and try to handle them accordingly.

<!-- more -->

The idea is that by having an isolated modules handle edge cases correctly, they will work even better when put together in a pipeline. This will certainly minimize the amount of surprise encountered once the model is deployed.

Unit test also gave extra incentive to test by giving access to mocked input, means edge cases data can be artificially made without having to integrate with costly data sources (both in terms of money and compute resource).

This post explains how to set up unit testing environment and shows simple example of test cases, we are going to use [Scala API](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/index.html) of Spark and Maven as build tool.

## Environment Setup

First and foremost, we have to setup scalatest library in to our project. [Scalatest](https://www.scalatest.org/) supports many test styles, runners, and rich assertion library. It will make unit testing easier. We can setup scalatest using maven siply by adding the following lines to our `pom.xml`

```xml
<dependencies>
  ...
  <!--  other dependencies-->
  ...
  <dependency>
    <groupId>org.scalactic</groupId>
    <artifactId>scalactic_${scala.binary.version}</artifactId>
    <version>${scala.test.version}</version>
  </dependency>
  <dependency>
    <groupId>org.scalatest</groupId>
    <artifactId>scalatest-funsuite_${scala.binary.version}</artifactId>
    <version>${scala.test.version}</version>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>org.scalatest</groupId>
    <artifactId>scalatest-matchers-core_${scala.binary.version}</artifactId>
    <version>${scala.test.version}</version>
    <scope>test</scope>
  </dependency>
</dependencies>
```

Do take note on `scalatest-funsuite` part of the configuration. This is the test style that we are going to use. I will explain a bit more about test styles later, so just take a note of this part for now.

Once the `pom.xml` has been set up, compile the project to let maven refreshes. It usually done by running `mvn clean package` command unless you have setup a custom maven build process.

## Funsuite Test Styles

In this post, we use funsuite test style because it is the style that is used by Spark themselves, means if we are stuck and in need for inspiration, simply go to Spark project on github where it host tons of battle-tested test cases.

Now let's imagine a function that will binarize a column based on a threshold number. Any number less than this number will be converted into 0s and the rest will be converted into 1s.

```scala
package io.github.iahsanujunda.spark.unitTest.example

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object Binarizer {
  def process(threshold: Int, inputColName: String)(df: DataFrame): DataFrame = {
    df.withColumn(
        "output",
        when(inputColName < threshold, 0).otherwise(1) 
    )
  }
}
```

This function will work basically as follows.

```bash
// before
+-----+--------+
| id  | metric |
+-----+--------+
| id1 |   2    |
| id2 |   8    |
| id3 |   1    |
| id4 |   4    |
| id5 |   2    |
| id6 |   9    |
+-----+--------+

// with threshold = 5
+-----+--------+--------+
| id  | metric | output |
+-----+--------+--------|
| id1 |   2    |   0    |
| id2 |   8    |   1    |
| id3 |   1    |   0    |
| id4 |   4    |   0    |
| id5 |   2    |   0    |
| id6 |   9    |   1    |
+-----+--------+--------+
```

With funsuite style, test case for above scenario will look like this

```scala
package io.github.iahsanujunda.spark.unitTest.example

import org.scalatest.Matchers
import org.scalatest.funsuite.AnyFunSuite

class HelloWorldSuite extends AnyFunSuite with Matchers {
    
    import spark.implicits._

    test("must correctly identify 0s and 1s") {
        import io.github.iahsanujunda.spark.unitTest.example.Binarizer
        
        val mockedDf = Seq(
            ("id1", 2)
            ("id2", 8)
            ("id3", 1)
            ("id4", 4)
            ("id5", 2)
            ("id6", 9)
        ).toDF("id", "metric")
 
        val result = mockedDf.transform(Binarizer.process(5, "metric"))
        val zeros  = result.filter("output = 0").select("id).collect.map(_(0))
        val ones   = result.filter("output = 1").select("id).collect.map(_(0))
        zeros should contain allOf ("id1", "id3", "id4", "id5")
        ones should contain allOf ("id2", "id6")
    }
}
```

Easy isn't it? Funsuite and Matchers allows us to write an easy-to-read test cases.

Now let's say we want to add some extra functionality to our Binarizer. We want the output column to be configurable. Let's write the test case first.

```scala
...
test("must correctly produce configurable schema") {
    import io.github.iahsanujunda.spark.unitTest.example.Binarizer
    
    val mockedDf = Seq(
        ("id1", 2)
        ("id2", 8)
        ("id3", 1)
        ("id4", 4)
        ("id5", 2)
        ("id6", 9)
    ).toDF("id", "metric")

    val result = mockedDf.transform(Binarizer.process(5, "metric", "binarized_column"))
    val schema = result.schema.fieldNames 
    schema should contain ("binarized_column")
}
...
```

When we run it now, it will fail, so we have to modify our Binarizer implementation as well.

```scala
object Binarizer {
  def process(threshold: Int, inputColName: String, outputColName: String)(df: DataFrame): DataFrame = {
    df.withColumn(
        outputColName,
        when(inputColName < threshold, 1).otherwise(0) 
    )
  }
}
```

If we run the test case now, it will success. Wait... did we just casually summon TDD out of nowhere? Yes, we are. TDD is a very good software development practice. Considering we write our machine learning pipeline as a software, no harm in adding another jargon to our practice if it make our ML model better.

Now that we tackle the basic functionality, we can move on to an even richer test cases. What if we supply a column of strings as the input? Should we try to convert this string into number types if possible? or should we just throw an exception with a friendly advice to our fellow developer to please dont do that? What if we pass output column name that already exists in the dataframe? What if the threshold number appears as input, should it be 0 or should it be 1? We can think of lots and lots of edge cases, and we can be confident that when our model face these gangsters in production, it wont just pack up and run away.

If you need an inspiration on **fantastic tests and how to find them**, scalatest already wrote a comprehensive guide on the test styles and matcher objects. You can [check them out here](https://www.scalatest.org/at_a_glance/FunSuite).

## Other Test Styles

Remember what I mentioned previously, that scalatest support various test style? Funsuite is just one of them. If you come from python world and already familiar with `unittest` library, you might find [PropSpec](https://www.scalatest.org/at_a_glance/PropSpec) style familiar. For those of you who come from Javascript world, [FunSpec](https://www.scalatest.org/at_a_glance/FunSpec) might be more of your cup.

When you decide to use another test style other than funsuite, then you should go back to `pom.xml` file above, and locate the `scalatest-funsuite` line. Change the `funsuite` part into the style of your choosing.

In the next part, I would try to explain how to organize our Spark project to have a clean and maintainable Spark code base.

Thanks for reading! I hope you gain something useful from this.

Stay safe, stay healthy~