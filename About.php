<?php
include_once("Header.php");
include_once("DB_Files/db.php");
// include_once("./templates/base.php");
?>

<link rel="stylesheet" href="CSS/About.css">
<link rel="stylesheet" href="CSS/responsiveness.css">
<section class="about__achievements">
    <div class="container about__achievements-container">
        <div class="about__achievements-left">
            <img src="Img/about achievements.svg" alt="">
        </div>
        <div class="about__achievements-right">
            <h1>Achievements</h1>
            <p>Our global community and our course catalog get bigger every day.Check out our latest numbers as of December 2024.</p>
            <div class="achievements__cards">
                <article class="achievements__card">
                    <span class="achievement__icon">
                        <i class="uil uil-video"></i>
                    </span>
                    <h3>80+</h3>
                    <p>Courses</p>
                </article>

                <article class="achievements__card">
                    <span class="achievement__icon">
                        <i class="uil uil-users-alt"></i>
                    </span>
                    <h3>1500+</h3>
                    <p>Students</p>
                </article>

                <article class="achievements__card">
                    <span class="achievement__icon">
                        <i class="uil uil-trophy"></i>
                    </span>
                    <h3>8+</h3>
                    <p>Awards</p>
                </article>

            </div>
        </div>
    </div>
</section>
<section class="team reveal">
    <h2>Meet Our Team</h2>
    <div class="container team__container">
        <?php
       // $sql = "SELECT * FROM lectures where l_id=1 ";
       $sql = "SELECT * FROM lectures WHERE l_id IN (2, 10)";

        $result = $conn->query($sql);
        if ($result->num_rows > 0) {
            while ($row = $result->fetch_assoc()) {
                $l_id = $row['l_id'];
                echo '
        <article class="team__member">
            <div class="team__member-image">
                <img src="' . str_replace('..', '.', $row['l_img']) . '" alt="">
            </div>
            <div class="team__member-info">
                <h4>' . $row['l_name'] . '</h4>
                <p>' . $row['l_design'] . '</p>
            </div>
        </article>
        ';
            }
        }
        ?>
    </div>
</section>

<section class="faqs reveal">

    <h2>Frequently Asked Questions</h2>
    <div class="container faqs__container">
        <article class="faq">
            <div class="faq__icon"><i class="uil uil-plus"></i></div>
            <div class="question__answer ">
                <h4>What do E-LEARN courses include?</h4>
                <p>Each E-Learn course is created, owned and managed by the instructor(s). The foundation of each E-Learn course are its lectures, which can include videos. In addition, instructors can add resources and various types of practice activities, as a way to enhance the learning experience of students. </p>
            </div>
        </article>

        <article class="faq">
            <div class="faq__icon"><i class="uil uil-plus"></i></div>
            <div class="question__answer ">
                <h4>Is there any way to preview a course?</h4>
                <p>Yes! For steps on how to preview a course, and review key information about it.</p>
            </div>
        </article>

        <article class="faq">
            <div class="faq__icon"><i class="uil uil-plus"></i></div>
            <div class="question__answer ">
                <h4>How can I pay for a course?</h4>
                <p>E-Learn supports several different payment methods, depending on your account country and location. </p>
            </div>
        </article>

        <article class="faq">
            <div class="faq__icon"><i class="uil uil-plus"></i></div>
            <div class="question__answer ">
                <h4>What if I don’t like a course I purchased?</h4>
                <p>We want you to be satisfied, so all eligible courses purchased on E-Learn can be refunded within 30 days.</p>
            </div>
        </article>

        <article class="faq">
            <div class="faq__icon"><i class="uil uil-plus"></i></div>
            <div class="question__answer ">
                <h4>Where can I go for help?</h4>
                <p>If you find you have a question about a paid course while you’re taking it, you can search for answers to your question in the Q&A or ask the instructor. </p>
            </div>
        </article>

        <article class="faq">
            <div class="faq__icon"><i class="uil uil-plus"></i></div>
            <div class="question__answer ">
                <h4>Do I have to start my E-Learn course at a certain time? </h4>
                <p>If you find you have a question about a paid course while you are taking it, you can search for answers to your question in the Q&A or ask the instructor. </p>
            </div>
        </article>

        <article class="faq">
            <div class="faq__icon"><i class="uil uil-plus"></i></div>
            <div class="question__answer ">
                <h4>Do I have to start my E-Learn course at a certain time? And how long do I have to complete it?</h4>
                <p>As noted above, there are no deadlines to begin or complete the course. Even after you complete the course you will continue to have access to it, provided that your account is in good standing, and E-Learn continues to have a license to the course. </p>
            </div>
        </article>


    </div>
</section>


<script src="js/Custom.js"></script>

<script>
function reveal() {
    var reveals = document.querySelectorAll(".reveal");

    for (var i = 0; i < reveals.length; i++) {
        var windowHeight = window.innerHeight;
        var elementTop = reveals[i].getBoundingClientRect().top;
        var elementVisible = 150;

        if (elementTop < windowHeight - elementVisible) {
            reveals[i].classList.add("active");
        } else {
            reveals[i].classList.remove("active");
        }
    }
}

window.addEventListener("scroll", reveal);
</script>
<iframe allow="microphone" id="chatbot-frame" src="http://127.0.0.1:5000/" width="500" height="600" frameborder="0" style="position: fixed; bottom: 0px; right: 0px;"></iframe><?php
include_once("Footer.php");
?>