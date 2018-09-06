# Writer identification from handwriting using 3 approaches

In this project I used the 'AND' dataset which is labelled with the writer number who wrote that 'and' , The dataset is avalaible in image form of handwritten word 'and' and also in  human-engineered feature form (curvature of 'a' , slant of 'd' etc). We use 3 approaches to solve the problem of identifying the writer from the handwritting 
* PGM ( Mulitnet )
We use the human engineered features (curvature of 'a' , slant of 'd' etc) to build our two Bayesian Networks where one would represent the two features are from the same writer and the other would represent they are from diffrent writer. Using this knowledge of probability obtained from Bayesian Networks we can find out whether the two features belong to handwriting of the same
writer or not.
* Siamese Network with extracted SIFT features
I extracted SIFT features and from uniform sized features using bag-of-features method.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc






