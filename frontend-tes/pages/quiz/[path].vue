<!-- pages/quiz/[path].vue -->
<template>
    <div class="mx-auto py-[233px] max-w-[793px] font-montserrat">
        <div class="flex flex-col justify-center items-center gap-[73px] relative">
            <!-- Tombol Kembali -->
            <div class="absolute top-[-100px] left-0" v-if="currentIndex > 0">
                <button @click="prevQuestion" class="text-[#0099FF] text-[18px] font-bold underline">←
                    Sebelumnya</button>
            </div>

            <h1 class="font-semibold text-[30px] leading-[32px] text-center">{{ question }}</h1>

            <div class="flex flex-col gap-[23px] w-full text-[20px] leading-[14px] font-semibold text-[#333333]">
                <button :class="buttonClass('Ya')" @click="selectAnswer('Ya')">Ya</button>
                <button :class="buttonClass('Tidak')" @click="selectAnswer('Tidak')">Tidak</button>
            </div>
        </div>

        <!-- Popup akhir -->
        <div v-if="showResult"
            class="fixed z-10 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 max-w-[386px] max-h-[377px] bg-[#FDFDFD] border-[2px] border-[#00000040] rounded-[4px] font-semibold leading-[14px] text-[#000000] px-[53px] py-[86px] flex flex-col gap-[80px]">
            <p class="text-[24px]">Lihat Hasil Kamu</p>
            <button class="bg-[#0099FF] text-white shadow-[0_4px_40px_0_#00000026] rounded-[8px] text-[16px] py-[19px]"
                @click="submitAnswers">
                Lihat Hasil
            </button>
        </div>
    </div>
</template>

<script setup>
import { useRoute, useRouter } from 'vue-router'
import { ref, watch, onMounted } from 'vue'
import questions from '~/static/questions.json'
import { useAnswerStore } from '~/stores/answerStore'
import { useResultStore } from '~/stores/resultStore'

const route = useRoute()
const router = useRouter()

const answerStore = useAnswerStore()

// Inisialisasi jawaban global saat komponen mount
onMounted(() => {
    const index = parseInt(route.params.path) - 1

    // Load dari localStorage jika ada
    answerStore.initAnswers(questions.length)

    // Validasi agar user tidak bisa langsung ke soal tertentu
    if (!answerStore.allPreviousAnswered(index)) {
        alert("Jawab pertanyaan sebelumnya terlebih dahulu.")
        const redirectTo = answerStore.firstUnansweredIndex()
        router.replace(`/quiz/${redirectTo + 1}`)
    } else {
        currentIndex.value = index
    }
})

// Index dari path param (1-based ke 0-based)
const currentIndex = ref(parseInt(route.params.path) - 1)

// Validasi saat route berubah
watch(() => route.params.path, (newPath) => {
    const targetIndex = parseInt(newPath) - 1
    if (!answerStore.allPreviousAnswered(targetIndex)) {
        alert("Jawab pertanyaan sebelumnya terlebih dahulu.")
        const redirectTo = answerStore.firstUnansweredIndex()
        router.replace(`/quiz/${redirectTo + 1}`)
    } else {
        currentIndex.value = targetIndex
    }
})


// Pertanyaan aktif
const question = ref(questions[currentIndex.value])
watch(currentIndex, (val) => {
    question.value = questions[val]
})

// Pilih jawaban
const selectAnswer = (choice) => {
    const value = choice === 'Ya' ? 1 : 0
    answerStore.setAnswer(currentIndex.value, value)

    if (currentIndex.value + 1 < questions.length) {
        router.push(`/quiz/${currentIndex.value + 2}`)
    } else {
        showResult.value = true
    }
}

// Kembali ke soal sebelumnya
const prevQuestion = () => {
    if (currentIndex.value > 0) {
        router.push(`/quiz/${currentIndex.value}`)
    }
}

// Styling tombol berdasarkan jawaban
const buttonClass = (choice) => {
    const expected = choice === 'Ya' ? 1 : 0
    const selected = answerStore.getAnswer(currentIndex.value) === expected
    return [
        'py-[19px] text-center border border-[#FCBC1D] shadow-[0_4px_20px_0_#0000001A] rounded-[10px]',
        selected ? 'bg-[#FCBC1D] text-white' : ''
    ].join(' ')
}

const showResult = ref(false)

// Kirim jawaban
const submitAnswers = async () => {

    try {
        if (answerStore.answers.includes(null)) {
            alert("Jawaban belum lengkap.")
            return
        }
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                responses: answerStore.answers
            })
        })

        if (!response.ok) throw new Error('Gagal mendapatkan hasil dari API.')

        const result = await response.json()

        const resultStore = useResultStore()
        resultStore.setResult(result)
        answerStore.resetAnswers(0)

        router.push('/quiz/result')
    } catch (error) {
        console.error('Error saat submit:', error)
        alert('Gagal mengirim jawaban. Silakan coba lagi.')
    }
}
</script>
