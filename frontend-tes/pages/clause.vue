<template>
    <div class="flex flex-col justify-center items-center pt-[106px] pb-[129px]">
        <div
            class="font-montserrat rounded-[10px] bg-[#FEFEFE] border border-[#00000040] shadow-[0_4px_4px_0_#00000040] max-w-[796px] pt-[54px] px-[34px] flex flex-col">
            <div class="flex flex-col gap-[26px]">
                <h1 class="font-semibold text-xl leading-[40px] text-[#000000] text-center">
                    Cari Tahu Potensimu, Yuk!</h1>
                <div class="text-[16px] leading-[40px] tracking-[3%] text-[#000000]">
                    <p>Baca panduan pengisiannya ya!</p>
                    <ul class="list-disc pl-[20px]">
                        <li>Gak ada jawaban yang benar atau salah. Isilah dengan jujur sesuai kepribadianmu.</li>
                        <li>Santai aja, tes ini gak diberi waktu, kok.</li>
                        <li>Cari tempat yang nyaman dan kondusif supaya kamu lebih fokus.</li>
                        <li>Hasil tes bisa kamu dapatkan setelah mengisi semua pertanyaan dengan lengkap.</li>
                        <li>Tes minat belajar bantu kamu temukan jurusan yang cocok untuk kamu.</li>
                    </ul>
                </div>
            </div>
            <div class="py-[60px] flex gap-[20px] justify-center font-inter text-[#FAFAFA] text-[14px]">
                <NuxtLink to="/quiz/1" class="bg-[#0099FF] py-[19px] px-[30px] rounded-[8px]">Mulai Tes</NuxtLink>
                <NuxtLink to="/" class="bg-[#FCBC1D] py-[19px] px-[30px] rounded-[8px]">Kembali</NuxtLink>
            </div>
        </div>
    </div>
</template>