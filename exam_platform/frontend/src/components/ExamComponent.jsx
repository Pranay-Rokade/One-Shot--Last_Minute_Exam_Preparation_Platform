import { useState, useEffect } from "react";
import "./ExamComponent.css";
import axios from "axios";

function ExamComponent() {
  const [questions, setQuestions] = useState([]);
  const [userAnswers, setUserAnswers] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [results, setResults] = useState(null);

  useEffect(() => {
    fetchQuestions();
  }, []);

  const fetchQuestions = async () => {
    try {
      setLoading(true);
      const response = await axios.post("http://127.0.0.1:8000/generate-questions")

      if (!response.ok) {
        throw new Error('Failed to fetch questions');
      }

      const raw = await response.json();
      const data = JSON.parse(raw.questions);
      setQuestions(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleAnswerChange = (questionIndex, selectedOption) => {
    setUserAnswers({
      ...userAnswers,
      [questionIndex]: selectedOption,
    });
  };

  const handleSubmit = () => {
    let correctCount = 0;
    const resultDetails = questions.map((question, index) => {
      const userAnswer = userAnswers[index];
      const isCorrect = userAnswer === question.answer;

      if (isCorrect) {
        correctCount++;
      }

      return {
        question: question.question,
        options: question.options,
        userAnswer: userAnswer || "Not Answered",
        correctAnswer: question.answer,
        isCorrect: isCorrect,
      };
    });

    setResults({
      score: correctCount,
      total: questions.length,
      details: resultDetails,
    });
    setSubmitted(true);
  };

  const handleRetake = () => {
    setSubmitted(false);
    setUserAnswers({});
    setResults(null);
    fetchQuestions();
  };

  if (loading) {
    return (
      <div className="exam-container">
        <div className="loading">Loading questions...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="exam-container">
        <div className="error">Error: {error}</div>
        <button onClick={fetchQuestions} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  if (submitted && results) {
    return (
      <div className="exam-container">
        <div className="results-container">
          <h1>Exam Results</h1>
          <div className="score-card">
            <h2>
              Your Score: {results.score} / {results.total}
            </h2>
            <p className="percentage">
              {((results.score / results.total) * 100).toFixed(1)}%
            </p>
          </div>

          <div className="results-details">
            {results.details.map((result, index) => (
              <div
                key={index}
                className={`result-item ${
                  result.isCorrect ? "correct" : "incorrect"
                }`}
              >
                <div className="result-header">
                  <h3>Question {index + 1}</h3>
                  <span
                    className={`status-badge ${
                      result.isCorrect ? "correct" : "incorrect"
                    }`}
                  >
                    {result.isCorrect ? "✓ Correct" : "✗ Incorrect"}
                  </span>
                </div>

                <p className="question-text">{result.question}</p>

                <div className="options-list">
                  {result.options.map((option, optionIndex) => {
                    const optionLetter = option.charAt(0);
                    const isUserAnswer = result.userAnswer === optionLetter;
                    const isCorrectAnswer =
                      result.correctAnswer === optionLetter;

                    return (
                      <div
                        key={optionIndex}
                        className={`option-result ${
                          isCorrectAnswer ? "correct-answer" : ""
                        } ${
                          isUserAnswer && !isCorrectAnswer ? "wrong-answer" : ""
                        }`}
                      >
                        {option}
                        {isCorrectAnswer && (
                          <span className="label"> (Correct Answer)</span>
                        )}
                        {isUserAnswer && !isCorrectAnswer && (
                          <span className="label"> (Your Answer)</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          <button onClick={handleRetake} className="retake-btn">
            Retake Exam
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="exam-container">
      <h1>Exam - Answer All Questions</h1>
      <p className="exam-info">Total Questions: {questions.length}</p>

      <div className="questions-container">
        {questions.map((question, questionIndex) => (
          <div key={questionIndex} className="question-card">
            <h3>Question {questionIndex + 1}</h3>
            <p className="question-text">{question.question}</p>

            <div className="options">
              {question.options.map((option, optionIndex) => {
                const optionLetter = option.charAt(0);
                return (
                  <label key={optionIndex} className="option-label">
                    <input
                      type="radio"
                      name={`question-${questionIndex}`}
                      value={optionLetter}
                      checked={userAnswers[questionIndex] === optionLetter}
                      onChange={() =>
                        handleAnswerChange(questionIndex, optionLetter)
                      }
                    />
                    <span className="option-text">{option}</span>
                  </label>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="submit-section">
        <button
          onClick={handleSubmit}
          className="submit-btn"
          disabled={Object.keys(userAnswers).length !== questions.length}
        >
          Submit Exam
        </button>
        {Object.keys(userAnswers).length !== questions.length && (
          <p className="warning">
            Please answer all questions before submitting
          </p>
        )}
      </div>
    </div>
  );
}

export default ExamComponent;
